# Qwen GRPO Trainer

This is a compact, from-scratch GRPO training example for `Qwen/Qwen3-0.6B`.
It deliberately does not use `trl.GRPOTrainer`, so the important parts of the
algorithm are visible in one file:

- sample a group of completions for each prompt
- score each completion with a reward function
- normalize rewards within each prompt group to get relative advantages
- optimize a clipped policy objective with a reference-model KL penalty

## Install

```bash
cd ~/workspace/repository/infiniai/grpo
export HF_ENDPOINT=https://hf-mirror.com
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Data Format

Use JSONL. Each row needs a `prompt`. For the built-in exact-match reward,
also provide `answer`.

```json
{"prompt":"What is 2 + 3? Return only the number.","answer":"5"}
{"prompt":"Compute 12 * 7. Return only the number.","answer":"84"}
```

See `examples/toy_math.jsonl`.

## Run

```bash
export HF_ENDPOINT=https://hf-mirror.com

python src/train_grpo.py \
  --dataset examples/toy_math.jsonl \
  --model_name Qwen/Qwen3-0.6B \
  --output_dir outputs/qwen3-0.6b-grpo \
  --group_size 4 \
  --per_device_prompts 2 \
  --max_prompt_length 256 \
  --max_new_tokens 128 \
  --num_steps 100
```

For a small GPU, start with `--group_size 2 --per_device_prompts 1` and keep
`--gradient_checkpointing`.

The trainer prints timestamped JSON logs at key nodes: startup, tokenizer/model
loading, dataset loading, rollout generation, loss computation, optimizer step,
metrics, sample completions, and checkpoint saves. Use `--log_every` to control
metric frequency and `--sample_log_every 0` to disable sample completion logs.

## Notes

- The default reward is intentionally simple. Replace `reward_exact_match` or
  add your own reward function for real tasks.
- GRPO still needs a reward signal. The difference from PPO is that the
  advantage baseline comes from the group of completions for the same prompt,
  not from a learned value model.
- The script keeps a frozen reference model for KL regularization.
