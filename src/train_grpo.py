"""
GRPO (Group Relative Policy Optimization) 训练脚本
====================================================

## 算法原理

GRPO 是专为语言模型 RLHF 设计的强化学习算法，在 PPO 基础上用组内相对奖励替代 Critic 网络，
大幅降低显存消耗，同时保留 PPO 的 clipped surrogate 稳定性。

### 工作流程

    ┌─────────────────────────────────────────────────────────────────┐
    │  for each training step:                                        │
    │                                                                 │
    │  1. Rollout   对每条 prompt 用 π_θ 采样 G 条补全               │
    │  2. Reward    对每条补全打分得 r_1 … r_G                        │
    │  3. Advantage 组内 z-score 归一化，得优势 A_1 … A_G            │
    │  4. Update    num_ppo_epochs 轮：最大化 GRPO 目标，梯度更新 θ   │
    └─────────────────────────────────────────────────────────────────┘

### 核心公式

**Step 3 — 组内优势归一化（替代 Critic）：**

    A_i = (r_i − mean(r_{1..G})) / (std(r_{1..G}) + ε)

**Step 4a — 概率比率：**

    ρ_t = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)
        = exp( log π_θ(a_t|s_t) − log π_θ_old(a_t|s_t) )

**Step 4b — Clipped Surrogate Objective（PPO-Clip）：**

    L_clip(t) = min( ρ_t · A_i,   clip(ρ_t, 1−ε, 1+ε) · A_i )

    ε（clip_eps）限制单步更新幅度：当 ratio 超出 [1−ε, 1+ε] 时梯度被截断，
    防止策略单步跳变过大。

**Step 4c — KL 散度惩罚（K3 近似，恒非负）：**

    KL_approx(t) = exp(Δ_t) − Δ_t − 1,   Δ_t = log π_ref(a_t|s_t) − log π_θ(a_t|s_t)

    在 Δ=0 处退化为 0，是真实 KL(π_θ ‖ π_ref) 的下界近似，
    β（beta）控制其对策略偏离参考模型的惩罚力度。

**Step 4d — 最终每 token 损失（最小化）：**

    L = − (1 / |completion_tokens|) · Σ_t [ L_clip(t) − β · KL_approx(t) ]

---

## 脚本功能

对 CausalLM（默认 Qwen/Qwen3-0.6B）做 GRPO 强化微调，奖励信号来自对答案的精确匹配：
数值近似相等得 1.0，字符串包含匹配得 1.0，否则 0.0。

数据集为 JSONL 格式，每行须包含 `prompt` 字段，`answer` 字段可选（缺失时奖励恒为 0）。

---

## 使用方式

    python src/train_grpo.py \\
        --dataset        path/to/data.jsonl   \\
        --model_name     Qwen/Qwen3-0.6B      \\
        --output_dir     outputs/my-model     \\
        --num_steps      100                  \\
        --group_size     4                    \\  # G：每条 prompt 采样几条补全
        --per_device_prompts 2               \\  # 每步处理几条 prompt
        --max_prompt_length  256             \\
        --max_new_tokens     128             \\
        --num_ppo_epochs     1               \\  # >1 时 clip 真正生效（完整 GRPO）
        --clip_eps       0.2                 \\  # ε
        --beta           0.04                \\  # β
        --bf16                                   # GPU 上启用 bfloat16

关键参数说明：
  --group_size G        G 越大优势估计越稳定，显存/时间随 G 线性增加
  --num_ppo_epochs N    默认 1（等价于 REINFORCE+KL）；设 N>1 可重复利用 rollout
  --clip_eps ε          PPO-Clip 幅度，典型值 0.1~0.2
  --beta β              KL 惩罚系数，典型值 0.01~0.1；设 0 退化为纯 policy gradient
"""
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
    """将 Sample 列表转换为包含 prompts 和 answers 的批次字典。"""
    return {
        "prompts": [s.prompt for s in samples],
        "answers": [s.answer for s in samples],
    }


def last_number(text: str) -> str | None:
    """从文本中提取最后一个数字字符串，忽略千位分隔符逗号。"""
    matches = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", text.replace(",", ""))
    if not matches:
        return None
    return matches[-1]


def reward_exact_match(completions: list[str], answers: list[str | None]) -> torch.Tensor:
    """计算每条补全的奖励：数值精确匹配得 1.0，字符串包含匹配得 1.0，否则 0.0。"""
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
    """将 prompt 格式化为模型输入，有 chat_template 时应用模板，否则原样返回。"""
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def repeat_by_group(values: Iterable[str | None], group_size: int) -> list[str | None]:
    """将序列中每个元素重复 group_size 次，用于将答案与多组采样对齐。"""
    repeated: list[str | None] = []
    for value in values:
        repeated.extend([value] * group_size)
    return repeated


def setup_logging() -> None:
    """初始化结构化日志，输出到 stdout，格式含时间戳和级别。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def log_event(event: str, **fields: object) -> None:
    """以 JSON 格式记录一条带有 event 字段的结构化日志。"""
    payload = {"event": event, **fields}
    logger.info(json.dumps(payload, ensure_ascii=False, default=str))


def selective_log_softmax(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """对 logits 做 log_softmax 后，仅取 input_ids 对应位置的对数概率。"""
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)


def get_token_logprobs(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """前向传播并返回每个 next-token 位置的对数概率，形状 (batch, seq_len-1)。"""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    # logits[:, t] predicts input_ids[:, t + 1]
    return selective_log_softmax(outputs.logits[:, :-1, :], input_ids[:, 1:])


def build_attention_mask(
    generated: torch.Tensor,
    prompt_sequence_length: int,
    pad_token_id: int,
    eos_token_id: int,
) -> torch.Tensor:
    """构建生成序列的 attention mask，正确处理 pad_token_id == eos_token_id 的情况。

    当两者相同时，token-id 方式会把合法的 stop-EOS 也屏蔽掉，导致 EOS 动作无梯度信号。
    此函数找到每行补全部分第一个 EOS 的位置，保留该位置为 attended，之后置 0。
    """
    if pad_token_id != eos_token_id:
        return generated.ne(pad_token_id).long()

    mask = torch.ones(generated.shape, dtype=torch.long, device=generated.device)
    for i in range(generated.shape[0]):
        completion = generated[i, prompt_sequence_length:]
        eos_pos = (completion == eos_token_id).nonzero(as_tuple=False)
        if eos_pos.numel() > 0:
            first_eos = int(eos_pos[0, 0])
            mask[i, prompt_sequence_length + first_eos + 1 :] = 0
    return mask


def make_completion_mask(
    attention_mask: torch.Tensor,
    prompt_sequence_lengths: torch.Tensor,
) -> torch.Tensor:
    """生成布尔掩码，标记 next-token logprob 序列中属于补全部分的有效位置。"""
    # Mask is aligned with next-token logprobs, so position j corresponds to input_ids[:, j + 1].
    batch, seq_len_minus_1 = attention_mask[:, 1:].shape
    positions = torch.arange(seq_len_minus_1, device=attention_mask.device).unsqueeze(0).expand(batch, -1)
    completion_mask = positions >= (prompt_sequence_lengths.unsqueeze(1) - 1)
    return completion_mask & attention_mask[:, 1:].bool()


def grpo_advantages(rewards: torch.Tensor, group_size: int, eps: float = 1e-4) -> torch.Tensor:
    """对每组奖励做组内 z-score 归一化，返回扁平化的优势值张量。"""
    # rewards shape: (batch * G,)，将其重排为 (batch, G) 做组内统计
    # 公式：A_i = (r_i − mean(r_{1..G})) / (std(r_{1..G}) + ε)
    # 组内均值消除奖励绝对量级，除以标准差使优势量级稳定；
    # 若一组内所有奖励相同（std=0），A_i 全为 0，该步不产生策略梯度信号。
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
    """计算 GRPO 损失：clipped policy objective 减去 KL 惩罚项，返回损失及诊断指标。"""
    logprobs = get_token_logprobs(model, input_ids, attention_mask)
    with torch.no_grad():
        ref_logprobs = get_token_logprobs(ref_model, input_ids, attention_mask)

    # ρ_t = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) = exp(log π_θ − log π_θ_old)
    # 当参数尚未更新时 ρ_t ≈ 1；随 PPO epoch 推进 ρ_t 开始偏离，clip 才真正生效
    ratio = torch.exp(logprobs - old_logprobs)

    # advantages: (batch*G,) → (batch*G, 1) 以便与 per-token ratio (batch*G, T) 广播
    advantages = advantages.unsqueeze(1)

    # L_clip(t) = min( ρ_t · A_i,   clip(ρ_t, 1−ε, 1+ε) · A_i )
    # 当 A_i > 0（好于组均值）：若 ρ_t > 1+ε 则梯度截断，防止过度追加概率
    # 当 A_i < 0（差于组均值）：若 ρ_t < 1−ε 则梯度截断，防止过度削减概率
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_objective = torch.minimum(unclipped, clipped)

    # KL 散度 K3 近似：KL(π_θ ‖ π_ref) ≈ exp(Δ) − Δ − 1，Δ = log π_ref − log π_θ
    # 性质：Δ=0 时值为 0；恒非负；是真实 KL 的下界；相比 log(π_ref/π_θ) 数值更稳定
    logp_diff = ref_logprobs - logprobs
    kl = torch.exp(logp_diff) - logp_diff - 1.0

    # L = −E_t[ L_clip(t) − β · KL(t) ]，对补全 token 做归一化平均
    # 负号：我们最小化损失，等价于最大化 clipped surrogate 并惩罚 KL 偏移
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
    """为每条 prompt 采样 group_size 条补全，计算奖励、优势及旧对数概率，返回完整 rollout 字典。"""
    formatted_prompts = [format_prompt(tokenizer, prompt) for prompt in prompts]
    encoded = tokenizer(
        formatted_prompts,
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
        return_tensors="pt",
    ).to(device)
    prompt_sequence_length = encoded["input_ids"].shape[1]

    # Bug 2 fix: eval mode during rollout so dropout is disabled
    # Bug 4 fix: re-enable KV cache for O(N) generation, restore afterward
    model.eval()
    use_cache_orig = model.config.use_cache
    model.config.use_cache = True
    try:
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
    finally:
        model.config.use_cache = use_cache_orig
        model.train()

    repeated_prompt_lengths = torch.full(
        (generated.shape[0],),
        fill_value=prompt_sequence_length,
        dtype=torch.long,
        device=device,
    )
    # Bug 1 fix: use position-based mask so EOS is not wrongly zeroed when pad_id == eos_id
    repeated_attention = build_attention_mask(
        generated, prompt_sequence_length, tokenizer.pad_token_id, tokenizer.eos_token_id
    )
    completion_mask = make_completion_mask(repeated_attention, repeated_prompt_lengths)

    completions: list[str] = []
    for sequence, prompt_len in zip(generated, repeated_prompt_lengths.tolist(), strict=True):
        completion_ids = sequence[prompt_len:]
        completions.append(tokenizer.decode(completion_ids, skip_special_tokens=True))

    repeated_answers = repeat_by_group(answers, group_size)
    # Step 2: 对每条补全打分，rewards shape: (batch * G,)
    # 奖励函数：数值精确匹配 → 1.0，字符串包含匹配 → 1.0，否则 → 0.0
    rewards = reward_exact_match(completions, repeated_answers).to(device)
    # Step 3: 组内 z-score 归一化，得到优势 A_i = (r_i − mean) / (std + ε)
    advantages = grpo_advantages(rewards, group_size).to(device)
    # old_logprobs: rollout 时策略 π_θ_old 的 per-token 对数概率，用于后续计算 ρ_t
    # 必须在 eval 模式下计算，确保 dropout 关闭，与 grpo_loss 中的 logprobs 对比有意义
    model.eval()
    try:
        old_logprobs = get_token_logprobs(model, generated, repeated_attention).detach()
    finally:
        model.train()

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
    """解析命令行参数，涵盖模型、数据集、训练超参数及采样配置。"""
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
    parser.add_argument("--num_ppo_epochs", type=int, default=1,
                        help="Inner PPO epochs per rollout. >1 lets the clipped surrogate take effect.")
    return parser.parse_args()


def main() -> None:
    """训练入口：加载模型与数据集，执行 GRPO 训练循环，定期保存检查点。"""
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
        for ppo_epoch in range(args.num_ppo_epochs):
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
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        # Step LR scheduler once per rollout, not per inner PPO epoch
        scheduler.step()
        log_event(
            "step_loss_optim_done",
            step=step,
            elapsed_seconds=round(time.perf_counter() - loss_start, 4),
            grad_norm=float(grad_norm.detach().cpu()),
            lr=scheduler.get_last_lr()[0],
            **metrics,
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
