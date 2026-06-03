from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


logger = logging.getLogger("grpo")


@dataclass
class Sample:
    prompt: str
    answer: str | None = None


class JsonlPromptDataset(Dataset[Sample]):
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.rows: list[Sample] = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                obj = json.loads(line)
                prompt = obj.get("prompt")
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError(f"Row {line_no} must contain a non-empty string prompt")
                answer = obj.get("answer")
                self.rows.append(Sample(prompt=prompt, answer=str(answer) if answer is not None else None))
        if not self.rows:
            raise ValueError(f"No training rows found in {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Sample:
        return self.rows[idx]


def collate(samples: list[Sample]) -> dict[str, list[str | None]]:
    return {
        "prompts": [s.prompt for s in samples],
        "answers": [s.answer for s in samples],
    }


def last_number(text: str) -> str | None:
    matches = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", text.replace(",", ""))
    if not matches:
        return None
    return matches[-1]


def reward_exact_match(completions: list[str], answers: list[str | None]) -> torch.Tensor:
    rewards: list[float] = []
    for completion, answer in zip(completions, answers, strict=True):
        if answer is None:
            rewards.append(0.0)
            continue

        pred_num = last_number(completion)
        gold_num = last_number(answer)
        if pred_num is not None and gold_num is not None:
            rewards.append(1.0 if math.isclose(float(pred_num), float(gold_num), rel_tol=0, abs_tol=1e-6) else 0.0)
            continue

        normalized_completion = completion.strip().lower()
        normalized_answer = answer.strip().lower()
        rewards.append(1.0 if normalized_answer and normalized_answer in normalized_completion else 0.0)
    return torch.tensor(rewards, dtype=torch.float32)


def format_prompt(tokenizer: AutoTokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def repeat_by_group(values: Iterable[str | None], group_size: int) -> list[str | None]:
    repeated: list[str | None] = []
    for value in values:
        repeated.extend([value] * group_size)
    return repeated


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def log_event(event: str, **fields: object) -> None:
    payload = {"event": event, **fields}
    logger.info(json.dumps(payload, ensure_ascii=False, default=str))


def selective_log_softmax(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)


def get_token_logprobs(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    # logits[:, t] predicts input_ids[:, t + 1]
    return selective_log_softmax(outputs.logits[:, :-1, :], input_ids[:, 1:])


def make_completion_mask(
    attention_mask: torch.Tensor,
    prompt_sequence_lengths: torch.Tensor,
) -> torch.Tensor:
    # Mask is aligned with next-token logprobs, so position j corresponds to input_ids[:, j + 1].
    batch, seq_len_minus_1 = attention_mask[:, 1:].shape
    positions = torch.arange(seq_len_minus_1, device=attention_mask.device).unsqueeze(0).expand(batch, -1)
    completion_mask = positions >= (prompt_sequence_lengths.unsqueeze(1) - 1)
    return completion_mask & attention_mask[:, 1:].bool()


def grpo_advantages(rewards: torch.Tensor, group_size: int, eps: float = 1e-4) -> torch.Tensor:
    grouped = rewards.view(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True, unbiased=False)
    return ((grouped - mean) / (std + eps)).view(-1)


def grpo_loss(
    model: AutoModelForCausalLM,
    ref_model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float,
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    logprobs = get_token_logprobs(model, input_ids, attention_mask)
    with torch.no_grad():
        ref_logprobs = get_token_logprobs(ref_model, input_ids, attention_mask)

    ratio = torch.exp(logprobs - old_logprobs)
    advantages = advantages.unsqueeze(1)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_objective = torch.minimum(unclipped, clipped)

    # Non-negative per-token KL approximation used by several RLHF implementations.
    logp_diff = ref_logprobs - logprobs
    kl = torch.exp(logp_diff) - logp_diff - 1.0

    per_token_loss = -(policy_objective - beta * kl)
    denom = completion_mask.sum().clamp_min(1)
    loss = (per_token_loss * completion_mask).sum() / denom

    with torch.no_grad():
        clip_fraction = ((ratio - 1.0).abs() > clip_eps).float()
        metrics = {
            "loss": float(loss.detach().cpu()),
            "kl": float(((kl * completion_mask).sum() / denom).detach().cpu()),
            "clip_fraction": float(((clip_fraction * completion_mask).sum() / denom).detach().cpu()),
        }
    return loss, metrics


@torch.no_grad()
def sample_groups(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    answers: list[str | None],
    group_size: int,
    max_prompt_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> dict[str, torch.Tensor | list[str]]:
    formatted_prompts = [format_prompt(tokenizer, prompt) for prompt in prompts]
    encoded = tokenizer(
        formatted_prompts,
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
        return_tensors="pt",
    ).to(device)
    prompt_sequence_length = encoded["input_ids"].shape[1]

    generated = model.generate(
        **encoded,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        num_return_sequences=group_size,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    repeated_prompt_lengths = torch.full(
        (generated.shape[0],),
        fill_value=prompt_sequence_length,
        dtype=torch.long,
        device=device,
    )
    repeated_attention = generated.ne(tokenizer.pad_token_id).long()
    completion_mask = make_completion_mask(repeated_attention, repeated_prompt_lengths)

    completions: list[str] = []
    for sequence, prompt_len in zip(generated, repeated_prompt_lengths.tolist(), strict=True):
        completion_ids = sequence[prompt_len:]
        completions.append(tokenizer.decode(completion_ids, skip_special_tokens=True))

    repeated_answers = repeat_by_group(answers, group_size)
    rewards = reward_exact_match(completions, repeated_answers).to(device)
    advantages = grpo_advantages(rewards, group_size).to(device)
    old_logprobs = get_token_logprobs(model, generated, repeated_attention).detach()

    return {
        "input_ids": generated,
        "attention_mask": repeated_attention,
        "completion_mask": completion_mask,
        "old_logprobs": old_logprobs,
        "rewards": rewards,
        "advantages": advantages,
        "completions": completions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen/Qwen3-0.6B with a minimal GRPO loop.")
    parser.add_argument("--dataset", required=True, help="Path to JSONL data with prompt and optional answer fields.")
    parser.add_argument("--model_name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_dir", default="outputs/qwen3-0.6b-grpo")
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--per_device_prompts", type=int, default=2)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--sample_log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", help="Load models in bfloat16 when supported.")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 and device.type == "cuda" else torch.float32
    log_event(
        "startup",
        model_name=args.model_name,
        dataset=args.dataset,
        output_dir=args.output_dir,
        device=str(device),
        dtype=str(dtype),
        hf_endpoint=os.environ.get("HF_ENDPOINT"),
        num_steps=args.num_steps,
        per_device_prompts=args.per_device_prompts,
        group_size=args.group_size,
        max_prompt_length=args.max_prompt_length,
        max_new_tokens=args.max_new_tokens,
    )

    log_event("load_tokenizer_start", model_name=args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    log_event(
        "load_tokenizer_done",
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        has_chat_template=bool(getattr(tokenizer, "chat_template", None)),
    )

    log_event("load_policy_model_start", model_name=args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    log_event("load_policy_model_done")

    log_event("load_reference_model_start", model_name=args.model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    log_event(
        "load_reference_model_done",
        total_params=total_params,
        trainable_params=trainable_params,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        log_event("gradient_checkpointing_enabled")

    log_event("load_dataset_start", dataset=args.dataset)
    dataset = JsonlPromptDataset(args.dataset)
    log_event("load_dataset_done", num_rows=len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=args.per_device_prompts,
        shuffle=True,
        collate_fn=collate,
        drop_last=False,
    )
    data_iter = iter(loader)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.num_steps,
    )

    model.train()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_event("train_start")

    for step in range(1, args.num_steps + 1):
        step_start = time.perf_counter()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        log_event(
            "step_rollout_start",
            step=step,
            prompts=len(batch["prompts"]),
            completions=len(batch["prompts"]) * args.group_size,
        )
        rollout_start = time.perf_counter()
        rollout = sample_groups(
            model=model,
            tokenizer=tokenizer,
            prompts=batch["prompts"],
            answers=batch["answers"],
            group_size=args.group_size,
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )
        rollout_seconds = time.perf_counter() - rollout_start
        completion_tokens = int(rollout["completion_mask"].sum().detach().cpu())
        sequence_length = int(rollout["input_ids"].shape[1])
        log_event(
            "step_rollout_done",
            step=step,
            rollout_seconds=round(rollout_seconds, 4),
            sequence_length=sequence_length,
            completion_tokens=completion_tokens,
        )

        loss_start = time.perf_counter()
        loss, metrics = grpo_loss(
            model=model,
            ref_model=ref_model,
            input_ids=rollout["input_ids"],
            attention_mask=rollout["attention_mask"],
            completion_mask=rollout["completion_mask"],
            old_logprobs=rollout["old_logprobs"],
            advantages=rollout["advantages"],
            clip_eps=args.clip_eps,
            beta=args.beta,
        )
        log_event(
            "step_loss_done",
            step=step,
            loss_seconds=round(time.perf_counter() - loss_start, 4),
            **metrics,
        )

        optim_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        log_event(
            "step_optim_done",
            step=step,
            optim_seconds=round(time.perf_counter() - optim_start, 4),
            grad_norm=float(grad_norm.detach().cpu()),
            lr=scheduler.get_last_lr()[0],
        )

        if step % args.log_every == 0:
            rewards = rollout["rewards"].detach().float().cpu()
            advantages = rollout["advantages"].detach().float().cpu()
            lr = scheduler.get_last_lr()[0]
            log_event(
                "step_metrics",
                step=step,
                lr=lr,
                reward_mean=float(rewards.mean()),
                reward_min=float(rewards.min()),
                reward_max=float(rewards.max()),
                reward_std=float(rewards.std(unbiased=False)),
                advantage_mean=float(advantages.mean()),
                advantage_min=float(advantages.min()),
                advantage_max=float(advantages.max()),
                advantage_std=float(advantages.std(unbiased=False)),
                step_seconds=round(time.perf_counter() - step_start, 4),
                **metrics,
            )

        if args.sample_log_every > 0 and step % args.sample_log_every == 0:
            completions = rollout["completions"]
            rewards = rollout["rewards"].detach().float().cpu().tolist()
            log_event(
                "sample_completion",
                step=step,
                prompt=batch["prompts"][0],
                answer=batch["answers"][0],
                completion=completions[0][:500],
                reward=rewards[0],
            )

        if step % args.save_every == 0 or step == args.num_steps:
            ckpt_dir = output_dir / f"step-{step}"
            log_event("checkpoint_save_start", step=step, path=str(ckpt_dir))
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            log_event("checkpoint_save_done", step=step, path=str(ckpt_dir))

    log_event("final_save_start", path=str(output_dir))
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    log_event("final_save_done", path=str(output_dir))


if __name__ == "__main__":
    main()
