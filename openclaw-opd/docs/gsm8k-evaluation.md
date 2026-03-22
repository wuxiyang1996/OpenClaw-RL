# GSM8K Evaluation: Easy & Hard

Evaluate trained LoRA checkpoints on two GSM8K difficulty levels.

---

## Datasets

| Difficulty | HuggingFace Repo | Problems | Description |
|------------|-----------------|----------|-------------|
| **easy** | [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) (test split) | 1 319 | Original grade-school math problems (2–8 steps, basic arithmetic) |
| **hard** | [`reasoning-machines/gsm-hard`](https://huggingface.co/datasets/reasoning-machines/gsm-hard) (train split) | 1 319 | Same questions with numbers replaced by larger, less common values |

GSM-Hard was introduced in the [PAL paper](https://arxiv.org/abs/2211.10435)
(Gao et al., 2022). It tests whether a model can reason through the problem
structure rather than pattern-match on commonly seen number ranges. A model
that truly reasons should maintain similar accuracy on both; a large drop from
easy → hard suggests memorization or shortcut reliance.

### Field mapping

| | `openai/gsm8k` | `reasoning-machines/gsm-hard` |
|---|---|---|
| Question field | `question` | `input` |
| Answer field | `answer` (contains `####` separator) | `target` (plain number) |
| Split | `test` | `train` |

The eval script handles both formats automatically via the `--difficulty` flag.

---

## Scripts

### Quick start

```bash
conda activate openclaw-opd-lora

# Easy (original GSM8K)
bash openclaw-opd/run_gsm8k_easy_eval.sh

# Hard (GSM-Hard, larger numbers)
bash openclaw-opd/run_gsm8k_hard_eval.sh
```

### Run both in parallel

```bash
# Two background jobs — each launches its own SGLang server on a different port
PORT=30050 bash openclaw-opd/run_gsm8k_easy_eval.sh &
PORT=30051 bash openclaw-opd/run_gsm8k_hard_eval.sh &
wait
```

### Against a running training server (no extra GPU needed)

During dual-LoRA training, the SGLang engine is already serving both adapters.
Point the eval scripts at it directly:

```bash
# Evaluate adapter A (port 30000) on easy GSM8K
API_BASE=http://localhost:30000/v1 bash openclaw-opd/run_gsm8k_easy_eval.sh &

# Evaluate adapter B (port 30001) on hard GSM8K
API_BASE=http://localhost:30001/v1 bash openclaw-opd/run_gsm8k_hard_eval.sh &

wait
```

### Evaluate a specific checkpoint

```bash
# Single-LoRA checkpoint
ADAPTER_PATH=/workspace/OpenClaw-RL/ckpt/qwen3-4b-openclaw-opd-topk-lora/iter_0000005/model \
  bash openclaw-opd/run_gsm8k_hard_eval.sh

# Dual-LoRA checkpoint (evaluate lora_a on easy, lora_b on hard)
ADAPTER_PATH=/workspace/OpenClaw-RL/ckpt/qwen3-4b-openclaw-opd-topk-dual-lora/iter_0000003/model/lora_a \
  bash openclaw-opd/run_gsm8k_easy_eval.sh &
ADAPTER_PATH=/workspace/OpenClaw-RL/ckpt/qwen3-4b-openclaw-opd-topk-dual-lora/iter_0000003/model/lora_b \
  bash openclaw-opd/run_gsm8k_hard_eval.sh &
wait
```

---

## Environment Variables

All variables from `run_gsm8k_eval.sh` are supported. The easy/hard wrappers
just set `DIFFICULTY` and `OUTPUT` for you.

| Variable | Default | Description |
|----------|---------|-------------|
| `DIFFICULTY` | `easy` | `easy` or `hard` |
| `HF_CKPT` | `models/Qwen3-4B` | Base model path |
| `CKPT_DIR` | `ckpt/qwen3-4b-openclaw-opd-topk-lora` | Checkpoint directory (auto-detects latest) |
| `ADAPTER_PATH` | (empty) | Specific adapter path (overrides auto-detect) |
| `MERGED_MODEL` | (empty) | Already-merged model (skips LoRA merge) |
| `API_BASE` | (empty) | External server URL (skips launching SGLang) |
| `PORT` | `30050` | Port for the eval SGLang server |
| `TP` | `1` | Tensor parallelism |
| `NUM_PROBLEMS` | `1319` | Number of problems (both sets have 1319) |
| `CONCURRENCY` | `32` | Parallel request count |
| `TEMPERATURE` | `0.0` | Greedy decoding by default |
| `MAX_TOKENS` | `4096` | Max generation tokens |
| `OUTPUT` | `results/gsm8k_{difficulty}_eval_results.json` | Results file |

---

## Output

Results are saved as JSON:

```json
{
  "model_path": "/path/to/merged_model",
  "difficulty": "hard",
  "dataset": "GSM-Hard (large-number variant)",
  "num_problems": 1319,
  "num_correct": 842,
  "accuracy": 0.6384,
  "temperature": 0.0,
  "results": [
    {
      "index": 0,
      "question": "...",
      "ground_truth": "12345",
      "predicted": "12345",
      "correct": true,
      "response": "..."
    }
  ]
}
```

Default output paths:
- Easy: `openclaw-opd/results/gsm8k_easy_eval_results.json`
- Hard: `openclaw-opd/results/gsm8k_hard_eval_results.json`

---

## How It Works

1. **Model resolution**: merges LoRA adapter into base model (or uses `--api-base` for an already-running server)
2. **Server launch**: starts SGLang with the merged model (skipped when using `--api-base`)
3. **Dataset load**: fetches from HuggingFace (or local JSON), selects fields based on difficulty
4. **Concurrent evaluation**: sends problems via OpenAI-compatible API with configurable concurrency
5. **Grading**: extracts `\boxed{}` answer, normalizes, compares to ground truth
6. **Report**: prints accuracy summary and saves detailed JSON

---

## File Layout

```text
openclaw-opd/
├── eval_gsm8k.py                # Core evaluation logic (supports --difficulty easy|hard)
├── run_gsm8k_eval.sh            # Base eval script (DIFFICULTY env var)
├── run_gsm8k_easy_eval.sh       # Wrapper: DIFFICULTY=easy
├── run_gsm8k_hard_eval.sh       # Wrapper: DIFFICULTY=hard
└── results/
    ├── gsm8k_easy_eval_results.json   # Easy results (auto-created)
    └── gsm8k_hard_eval_results.json   # Hard results (auto-created)
```
