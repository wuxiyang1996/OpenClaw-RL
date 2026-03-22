"""
GSM8K evaluation for OpenClaw-OPD trained models (LoRA or merged).

Loads a trained model checkpoint, serves it with SGLang, runs GSM8K test
problems, grades answers, and reports accuracy.

Usage (LoRA checkpoint):
    python eval_gsm8k.py \
        --base-model /path/to/Qwen3-4B \
        --adapter /path/to/ckpt/iter_0000001/model \
        --num-problems 200

Usage (already-merged HF checkpoint):
    python eval_gsm8k.py \
        --model /path/to/merged_model \
        --num-problems 200

Usage (auto-detect latest LoRA checkpoint from training):
    python eval_gsm8k.py \
        --base-model /path/to/Qwen3-4B \
        --ckpt-dir /path/to/ckpt/qwen3-4b-openclaw-opd-topk-lora \
        --num-problems 200
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SLIME_ROOT = REPO_ROOT / "slime"
MATH_UTILS_DIR = SLIME_ROOT / "slime" / "rollout" / "rm_hub"

GSM8K_SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the problem step by step. "
    "Put your final numerical answer inside \\boxed{}."
)


def find_latest_checkpoint(ckpt_dir: str) -> Path:
    """Find the latest checkpoint iteration in a training checkpoint directory."""
    ckpt_path = Path(ckpt_dir)
    tracker = ckpt_path / "latest_checkpointed_iteration.txt"
    if tracker.exists():
        step = int(tracker.read_text().strip())
        model_dir = ckpt_path / f"iter_{step:07d}" / "model"
        if model_dir.exists():
            return model_dir
    iter_dirs = sorted(ckpt_path.glob("iter_*/model"), key=lambda p: p.parent.name)
    if iter_dirs:
        return iter_dirs[-1]
    raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")


def merge_lora_adapter(base_model: str, adapter: str, output: str) -> str:
    """Merge LoRA adapter into base model using the existing merge tool."""
    merge_script = SLIME_ROOT / "tools" / "merge_lora_adapter.py"
    if not merge_script.exists():
        raise FileNotFoundError(f"Merge tool not found at {merge_script}")

    print(f"Merging LoRA adapter into base model...")
    print(f"  Base model: {base_model}")
    print(f"  Adapter:    {adapter}")
    print(f"  Output:     {output}")

    cmd = [
        sys.executable, str(merge_script),
        "--base-model", base_model,
        "--adapter", adapter,
        "--output", output,
        "--force",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}", file=sys.stderr)
        raise RuntimeError(f"LoRA merge failed (exit code {result.returncode})")
    print(result.stdout)
    return output


HF_DATASET_REGISTRY: dict[str, dict] = {
    "easy": {
        "repo": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "question_field": "question",
        "answer_field": "answer",
        "answer_parse": lambda a: a.split("####")[-1].strip() if "####" in a else a,
        "label": "GSM8K (easy — original test set)",
    },
    "hard": {
        "repo": "reasoning-machines/gsm-hard",
        "config": None,
        "split": "train",
        "question_field": "input",
        "answer_field": "target",
        "answer_parse": lambda a: str(a).strip(),
        "label": "GSM-Hard (large-number variant)",
    },
}


def load_gsm8k_dataset(
    dataset_path: str | None,
    num_problems: int,
    fold_index: int = 0,
    num_folds: int = 1,
    difficulty: str = "easy",
) -> list[dict]:
    """Load GSM8K problems, optionally selecting a specific fold.

    ``difficulty`` selects the HuggingFace dataset variant:
      - ``"easy"``  → ``openai/gsm8k`` test split (1 319 problems)
      - ``"hard"``  → ``reasoning-machines/gsm-hard`` train split (1 319 problems,
        same questions with larger numbers)

    When num_folds > 1, the first ``num_problems`` items are split into
    ``num_folds`` contiguous chunks and only chunk ``fold_index`` is returned.
    """
    spec = HF_DATASET_REGISTRY.get(difficulty)
    if spec is None:
        raise ValueError(f"Unknown difficulty {difficulty!r}; choose from {list(HF_DATASET_REGISTRY)}")

    if dataset_path and Path(dataset_path).exists():
        print(f"Loading from local file: {dataset_path}")
        with open(dataset_path, encoding="utf-8") as f:
            data = json.load(f)
        q_key = spec["question_field"]
        a_key = spec["answer_field"]
        all_problems = []
        for item in data[:num_problems]:
            question = item.get(q_key) or item.get("question", "")
            answer = item.get(a_key) or item.get("answer", item.get("ground_truth_answer", ""))
            all_problems.append({
                "question": question,
                "answer": spec["answer_parse"](str(answer)),
            })
    else:
        print(f"Loading {spec['label']} from HuggingFace ({spec['repo']})...")
        try:
            from datasets import load_dataset
            load_args = [spec["repo"]]
            if spec["config"]:
                load_args.append(spec["config"])
            ds = load_dataset(*load_args, split=spec["split"])
        except Exception:
            local_json = SCRIPT_DIR / "GSM8K.json"
            if local_json.exists() and difficulty == "easy":
                print(f"HuggingFace unavailable, falling back to {local_json}")
                return load_gsm8k_dataset(str(local_json), num_problems, fold_index, num_folds, difficulty)
            raise RuntimeError(
                f"Cannot load {spec['repo']}: install `datasets` or provide --dataset path"
            )

        q_key = spec["question_field"]
        a_key = spec["answer_field"]
        all_problems = []
        for i, item in enumerate(ds):
            if i >= num_problems:
                break
            all_problems.append({
                "question": item[q_key],
                "answer": spec["answer_parse"](str(item[a_key])),
            })

    if num_folds > 1:
        n = len(all_problems)
        fold_size = n // num_folds
        start = fold_index * fold_size
        end = start + fold_size if fold_index < num_folds - 1 else n
        all_problems = all_problems[start:end]
        print(f"Fold {fold_index}/{num_folds}: problems [{start}, {end}) — {len(all_problems)} items")

    print(f"Loaded {len(all_problems)} problems ({difficulty})")
    return all_problems


def extract_boxed_answer(text: str) -> str | None:
    """Extract answer from \\boxed{...}."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    num_open = 0
    right_idx = None
    while i < len(text):
        if text[i] == "{":
            num_open += 1
        if text[i] == "}":
            num_open -= 1
            if num_open == 0:
                right_idx = i
                break
        i += 1

    if right_idx is None:
        return None

    boxed = text[idx:right_idx + 1]
    prefix = "\\boxed{"
    if boxed.startswith(prefix) and boxed.endswith("}"):
        return boxed[len(prefix):-1]
    prefix = "\\fbox{"
    if boxed.startswith(prefix) and boxed.endswith("}"):
        return boxed[len(prefix):-1]
    return None


def normalize_answer(answer: str) -> str:
    """Normalize a numerical answer for comparison."""
    answer = answer.strip()
    answer = answer.replace(",", "")
    answer = answer.replace("$", "")
    answer = answer.replace("%", "")
    answer = answer.strip()
    try:
        val = float(answer)
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return answer.lower().strip()


def grade_answer(predicted: str, ground_truth: str) -> bool:
    """Grade a predicted answer against ground truth."""
    sys.path.insert(0, str(MATH_UTILS_DIR.parent.parent))
    try:
        from slime.rollout.rm_hub.math_utils import grade_answer_verl
        return grade_answer_verl(predicted, ground_truth)
    except ImportError:
        pass

    pred_extracted = extract_boxed_answer(predicted)
    if pred_extracted is None:
        numbers = re.findall(r"-?\d[\d,]*\.?\d*", predicted)
        pred_extracted = numbers[-1] if numbers else ""

    return normalize_answer(pred_extracted) == normalize_answer(ground_truth)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks."""
    return re.sub(r"<think>[\s\S]*?</think>\s*", "", text).strip()


def wait_for_server(url: str, timeout: int = 300) -> bool:
    """Wait for SGLang server to become healthy."""
    print(f"Waiting for server at {url} ...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{url}/health", timeout=5)
            if resp.status_code == 200:
                print("Server is ready.")
                return True
        except requests.ConnectionError:
            pass
        time.sleep(3)
    return False


def launch_sglang_server(
    model_path: str,
    port: int = 30050,
    tp: int = 1,
    mem_fraction: float = 0.85,
    context_length: int = 32768,
) -> subprocess.Popen:
    """Launch an SGLang server and return the Popen handle."""
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--port", str(port),
        "--tp", str(tp),
        "--mem-fraction-static", str(mem_fraction),
        "--context-length", str(context_length),
        "--trust-remote-code",
        "--disable-radix-cache",
    ]
    env = os.environ.copy()
    env["SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK"] = "1"
    print(f"Launching SGLang server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc


def generate_response(
    base_url: str,
    question: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    model: str = "default",
) -> str:
    """Send a GSM8K question to the model and get a response."""
    messages = [
        {"role": "system", "content": GSM8K_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    if "/v1/v1/" in url:
        url = url.replace("/v1/v1/", "/v1/", 1)
    resp = requests.post(
        url,
        json=payload,
        timeout=180,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return strip_thinking(content)


def run_evaluation(args):
    model_path = args.model
    server_proc = None
    tmp_merged_dir = None

    try:
        # Step 1: Resolve model path & server
        base_url = args.api_base

        if base_url is not None:
            # External server — no need to merge or launch anything
            model_path = model_path or "(external server)"
            print(f"\nUsing external API: {base_url}")
        else:
            # Need a local model: merge LoRA if necessary
            if model_path is None:
                if args.adapter is None and args.ckpt_dir is not None:
                    adapter_path = find_latest_checkpoint(args.ckpt_dir)
                    print(f"Auto-detected latest checkpoint: {adapter_path}")
                elif args.adapter is not None:
                    adapter_path = Path(args.adapter)
                else:
                    print("ERROR: Provide --model, --adapter, --ckpt-dir, or --api-base", file=sys.stderr)
                    sys.exit(1)

                if args.base_model is None:
                    print("ERROR: --base-model required when using --adapter or --ckpt-dir", file=sys.stderr)
                    sys.exit(1)

                if args.merged_output:
                    merged_path = args.merged_output
                else:
                    tmp_merged_dir = tempfile.mkdtemp(prefix="gsm8k_merged_")
                    merged_path = tmp_merged_dir

                model_path = merge_lora_adapter(args.base_model, str(adapter_path), merged_path)

            print(f"\nModel path for evaluation: {model_path}")

            # Step 2: Launch SGLang server
            port = args.port
            server_proc = launch_sglang_server(
                model_path=model_path,
                port=port,
                tp=args.tp,
                mem_fraction=args.mem_fraction,
                context_length=args.context_length,
            )
            base_url = f"http://127.0.0.1:{port}"
            if not wait_for_server(base_url, timeout=args.server_timeout):
                print("ERROR: SGLang server failed to start.", file=sys.stderr)
                sys.exit(1)

        # Step 3: Load GSM8K dataset
        fold_index = getattr(args, "fold_index", 0)
        num_folds = getattr(args, "num_folds", 1)
        difficulty = getattr(args, "difficulty", "easy")
        problems = load_gsm8k_dataset(args.dataset, args.num_problems, fold_index, num_folds, difficulty)
        print(f"\nEvaluating on {len(problems)} GSM8K problems\n")

        # Step 4: Run evaluation (concurrent)
        correct = 0
        total = 0
        results = [None] * len(problems)
        concurrency = getattr(args, "concurrency", 32)
        print(f"  Concurrency: {concurrency}\n")

        def _eval_one(idx: int, problem: dict) -> dict:
            question = problem["question"]
            ground_truth = problem["answer"]
            try:
                response = generate_response(
                    base_url=base_url,
                    question=question,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    model=args.served_model_name,
                )
            except Exception as e:
                return {"index": idx, "correct": False, "error": str(e)}
            is_correct = grade_answer(response, ground_truth)
            extracted = extract_boxed_answer(response)
            return {
                "index": idx,
                "question": question,
                "ground_truth": ground_truth,
                "predicted": extracted,
                "correct": is_correct,
                "response": response,
            }

        done_count = 0
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_eval_one, i, p): i for i, p in enumerate(problems)}
            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()
                results[idx] = result
                done_count += 1
                total += 1
                if result.get("correct"):
                    correct += 1
                if "error" in result:
                    print(f"  [{done_count}/{len(problems)}] ERROR (#{idx}): {result['error']}")
                else:
                    status = "CORRECT" if result["correct"] else "WRONG"
                    print(
                        f"  [{done_count}/{len(problems)}] {status}  "
                        f"predicted={result.get('predicted')}  truth={result.get('ground_truth')}"
                    )

        results = [r for r in results if r is not None]

        # Step 5: Print summary
        accuracy = correct / total if total > 0 else 0.0
        diff_spec = HF_DATASET_REGISTRY.get(difficulty, {})
        diff_label = diff_spec.get("label", difficulty)
        print(f"\n{'='*60}")
        print(f"GSM8K Evaluation Results")
        print(f"{'='*60}")
        print(f"  Dataset:  {diff_label}")
        print(f"  Model:    {model_path}")
        print(f"  Problems: {total}")
        print(f"  Correct:  {correct}")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        print(f"{'='*60}")

        # Step 6: Save results
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = SCRIPT_DIR / "results" / "gsm8k_eval_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            "model_path": str(model_path),
            "difficulty": difficulty,
            "dataset": diff_label,
            "num_problems": total,
            "num_correct": correct,
            "accuracy": accuracy,
            "temperature": args.temperature,
            "results": results,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to {output_path}")

    finally:
        if server_proc is not None:
            print("\nShutting down SGLang server...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server_proc.kill()

        if tmp_merged_dir and os.path.exists(tmp_merged_dir):
            print(f"Cleaning up temp directory: {tmp_merged_dir}")
            shutil.rmtree(tmp_merged_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="GSM8K evaluation for OpenClaw-OPD trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    model_group = parser.add_argument_group("Model (pick one approach)")
    model_group.add_argument(
        "--model", type=str, default=None,
        help="Path to an already-merged HF model (skips LoRA merge)",
    )
    model_group.add_argument(
        "--base-model", type=str, default=None,
        help="Path to base HF model (e.g. Qwen3-4B)",
    )
    model_group.add_argument(
        "--adapter", type=str, default=None,
        help="Path to LoRA adapter directory (containing adapter_weights.pt)",
    )
    model_group.add_argument(
        "--ckpt-dir", type=str, default=None,
        help="Path to training checkpoint dir (auto-detects latest iteration)",
    )
    model_group.add_argument(
        "--merged-output", type=str, default=None,
        help="Where to save the merged model (default: temp dir, cleaned up after eval)",
    )

    server_group = parser.add_argument_group("Server")
    server_group.add_argument(
        "--api-base", type=str, default=None,
        help="Use an already-running OpenAI-compatible API (skip launching SGLang)",
    )
    server_group.add_argument("--port", type=int, default=30050)
    server_group.add_argument("--tp", type=int, default=1, help="Tensor parallelism")
    server_group.add_argument("--mem-fraction", type=float, default=0.85)
    server_group.add_argument("--context-length", type=int, default=32768)
    server_group.add_argument("--server-timeout", type=int, default=300)
    server_group.add_argument(
        "--served-model-name", type=str, default="default",
        help="Model name for the API request",
    )

    eval_group = parser.add_argument_group("Evaluation")
    eval_group.add_argument(
        "--difficulty", type=str, default="easy", choices=["easy", "hard"],
        help="Dataset variant: 'easy' = openai/gsm8k (original), 'hard' = reasoning-machines/gsm-hard (larger numbers)",
    )
    eval_group.add_argument(
        "--dataset", type=str, default=None,
        help="Path to local JSON file (overrides HuggingFace download)",
    )
    eval_group.add_argument("--num-problems", type=int, default=1319, help="Number of test problems (default: full test set)")
    eval_group.add_argument("--num-folds", type=int, default=1, help="Split problems into N folds (default: 1 = no split)")
    eval_group.add_argument("--fold-index", type=int, default=0, help="Which fold to evaluate (0-based, used with --num-folds)")
    eval_group.add_argument("--concurrency", type=int, default=32, help="Number of parallel requests (default: 32)")
    eval_group.add_argument("--temperature", type=float, default=0.0)
    eval_group.add_argument("--max-tokens", type=int, default=4096)
    eval_group.add_argument(
        "--output", type=str, default=None,
        help="Path to save results JSON (default: results/gsm8k_eval_results.json)",
    )

    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
