"""
audit/bfcl_regression_analysis.py

Reproduces the regression decomposition methodology from the audit report.
Decomposes the 50.3pp strict-score BFCL regression into:
  - Naming effect (strict vs loose scoring): 8.0pp. Measured. The fine-tune drops
    module prefixes; the base models show a 0pp strict-vs-loose gap.
  - Remainder: 42.3pp. This is an UPPER BOUND on the genuine behavioural regression.
    The failure-mode taxonomy below estimates a further ~7pp scorer parsing issue,
    so the genuine regression is best read as roughly 35 to 42pp.

The taxonomy shares are estimates, not measurements.

Usage:
    python audit/bfcl_regression_analysis.py [--results results/eval_results_redacted.json]

Redacted version: client identity and proprietary API schemas removed.
"""

import json
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audit.inference_stack_review import InferenceConfig, vram_budget, latency_validity


# ─── Failure mode taxonomy ────────────────────────────────────────────────────

FAILURE_MODES = {
    "module_prefix_drop": {
        "description": "Right function name, wrong namespace prefix (e.g. 'area_triangle' vs 'geometry.area_triangle')",
        "nature": "Partially artifact — strict scorer penalises; base model does not exhibit this",
        "estimated_share_pp": 8,   # measured: loose minus strict on the fine-tuned model
    },
    "parameter_extraction_failure": {
        "description": "Correct function name, incomplete or malformed arguments",
        "nature": "Genuine behavioral regression — model learned over-cautious ToolACE distribution",
        "estimated_share_pp": 20,
    },
    "outright_refusal": {
        "description": "Model declines to call or requests clarification instead of executing",
        "nature": "Genuine behavioral regression — ToolACE refusal examples contaminated training",
        "estimated_share_pp": 15,
    },
    "scorer_format_mismatch": {
        "description": "Model outputs correct call in <tool_call> XML; scorer partially misses due to format priority order",
        "nature": "Scorer artifact — evaluate_custom.py parsing order issue (corrected in v2)",
        "estimated_share_pp": 7,
    },
}


def decompose_regression(strict_score: float, loose_score: float, baseline_strict: float) -> dict:
    """
    Decompose a BFCL regression into artifact vs genuine components.

    Parameters
    ----------
    strict_score     : strict accuracy of fine-tuned model (exact name + params)
    loose_score      : loose accuracy of fine-tuned model (suffix name + params)
    baseline_strict  : strict accuracy of baseline model

    Returns
    -------
    dict with regression decomposition
    """
    total_regression = baseline_strict - strict_score
    prefix_artifact  = loose_score - strict_score          # name-match strictness artifact
    genuine_regression = total_regression - prefix_artifact

    parser_share = FAILURE_MODES["scorer_format_mismatch"]["estimated_share_pp"]
    return {
        "baseline_strict_pct":       round(baseline_strict * 100, 1),
        "finetuned_strict_pct":       round(strict_score * 100, 1),
        "finetuned_loose_pct":        round(loose_score * 100, 1),
        "total_regression_pp":        round(total_regression * 100, 1),
        "prefix_artifact_pp":         round(prefix_artifact * 100, 1),
        "genuine_regression_pp":      round(genuine_regression * 100, 1),
        "genuine_regression_low_pp":  round(genuine_regression * 100 - parser_share, 1),
        "artifact_share_pct":         round((prefix_artifact / total_regression) * 100, 1) if total_regression > 0 else 0,
        "genuine_share_pct":          round((genuine_regression / total_regression) * 100, 1) if total_regression > 0 else 0,
    }


def score_checkpoint_trajectory(checkpoints: list[dict]) -> str:
    """
    Assess whether a BFCL regression is training-duration-driven or data-driven.

    If accuracy at step 500 (~0.8 epochs) is within noise of the final checkpoint,
    the regression baked in before one full pass — implicating data distribution,
    not overfitting from extended training.

    Parameters
    ----------
    checkpoints : list of {"step": int, "strict_pct": float}

    Returns
    -------
    "data_distribution" | "training_duration" | "inconclusive"
    """
    if len(checkpoints) < 2:
        return "inconclusive"

    sorted_ckpts = sorted(checkpoints, key=lambda x: x["step"])
    first_score  = sorted_ckpts[0]["strict_pct"]
    final_score  = sorted_ckpts[-1]["strict_pct"]
    delta        = abs(final_score - first_score)

    if delta <= 3.0:
        return "data_distribution"
    elif delta >= 10.0:
        return "training_duration"
    else:
        return "inconclusive"


def evaluate_filtered_retrain(
    original_strict: float,
    filtered_strict: float,
    filtered_loose: float,
    baseline_strict: float,
    examples_removed: int,
    total_examples: int,
) -> dict:
    """
    Evaluate the impact of a targeted data filter (removing refusal examples)
    on BFCL accuracy.

    Parameters
    ----------
    original_strict   : strict accuracy before filtering
    filtered_strict   : strict accuracy after filtering
    filtered_loose    : loose accuracy after filtering
    baseline_strict   : baseline model strict accuracy (loose and strict are equal for the baseline)
    examples_removed  : number of training examples removed
    total_examples    : total training examples before filtering
    """
    recovery_pp     = filtered_strict - original_strict
    gap_to_baseline = baseline_strict - filtered_strict
    loose_gap       = baseline_strict - filtered_loose
    naming_gap      = filtered_loose - filtered_strict
    filter_pct      = (examples_removed / total_examples) * 100

    return {
        "original_strict_pct":   round(original_strict * 100, 1),
        "filtered_strict_pct":   round(filtered_strict * 100, 1),
        "filtered_loose_pct":    round(filtered_loose * 100, 1),
        "baseline_strict_pct":   round(baseline_strict * 100, 1),
        "recovery_pp":           round(recovery_pp * 100, 1),
        "gap_to_baseline_pp":    round(gap_to_baseline * 100, 1),
        "loose_gap_to_baseline_pp": round(loose_gap * 100, 1),
        "naming_gap_pp":         round(naming_gap * 100, 1),
        "examples_removed":      examples_removed,
        "filter_pct":            round(filter_pct, 1),
        # Under loose scoring the gap is small, so what remains is mostly module-prefix naming.
        "dominant_failure_mode": "module_prefix_naming" if loose_gap * 100 <= 10 else "mixed",
    }


def assess_inference_stack(cfg: InferenceConfig) -> dict:
    """Wrap the inference stack review so both scripts share one calculation."""
    vram = vram_budget(cfg)
    lat = latency_validity(cfg)
    return {
        "model_size_gb":               cfg.model_size_gb,
        "kv_cache_allocated_gb":       vram["kv_cache_allocated_gb"],
        "headroom_gb":                 vram["headroom_gb"],
        "kv_bytes_per_token":          vram["kv_bytes_per_token"],
        "peak_kv_demand_estimate_gb":  vram["peak_kv_demand_estimate_gb"],
        "oom_risk":                    vram["oom_risk"],
        "mem_fraction_recommendation": vram["recommended_mem_fraction"],
        "ttft_benchmark_p99_ms":       cfg.ttft_p99_ms,
        "ttft_production_low_ms":      lat["production_estimate_low_ms"],
        "ttft_production_high_ms":     lat["production_estimate_high_ms"],
        "ttft_production_method":      lat["estimate_method"],
        "throughput_tok_s":            cfg.throughput_tok_s,
    }


def print_report(decomp: dict, trajectory: str, retrain: dict, inference: dict) -> None:
    """Print a formatted audit summary to stdout."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich import box
        console = Console()
        _rich_report(console, decomp, trajectory, retrain, inference)
    except ImportError:
        _plain_report(decomp, trajectory, retrain, inference)


def _rich_report(console, decomp, trajectory, retrain, inference):
    from rich.table import Table
    from rich.panel import Panel
    from rich import box

    console.print("\n[bold blue]═══ LLM FINE-TUNING AUDIT — SUMMARY ═══[/bold blue]\n")

    # Regression decomposition
    t = Table(title="BFCL Regression Decomposition", box=box.SIMPLE_HEAVY, show_lines=True)
    t.add_column("Metric", style="bold")
    t.add_column("Value", justify="right")
    t.add_column("Nature")
    t.add_row("Baseline strict accuracy",  f"{decomp['baseline_strict_pct']}%",  "")
    t.add_row("Fine-tuned strict accuracy", f"{decomp['finetuned_strict_pct']}%", "[red]↓ regression[/red]")
    t.add_row("Fine-tuned loose accuracy",  f"{decomp['finetuned_loose_pct']}%",  "suffix-match relaxed")
    t.add_row("Total regression",           f"{decomp['total_regression_pp']}pp", "[red]critical[/red]")
    t.add_row("  — prefix artifact",        f"{decomp['prefix_artifact_pp']}pp",  "scorer design choice")
    t.add_row("  — genuine regression",     f"{decomp['genuine_regression_pp']}pp", "[red]behavioral change[/red]")
    console.print(t)

    # Checkpoint trajectory
    verdict = {
        "data_distribution": "[red]Data distribution — regression baked before epoch 1[/red]",
        "training_duration": "[yellow]Training duration — early stopping would help[/yellow]",
        "inconclusive":      "[dim]Inconclusive — more checkpoints needed[/dim]",
    }[trajectory]
    console.print(Panel(f"Checkpoint trajectory: {verdict}", title="Overfitting Analysis"))

    # Filtered retrain
    t2 = Table(title="Filtered Retrain Results", box=box.SIMPLE_HEAVY, show_lines=True)
    t2.add_column("Metric", style="bold")
    t2.add_column("Value", justify="right")
    t2.add_row("Original strict",    f"{retrain['original_strict_pct']}%")
    t2.add_row("Filtered strict",    f"{retrain['filtered_strict_pct']}%")
    t2.add_row("Filtered loose",     f"{retrain['filtered_loose_pct']}%")
    t2.add_row("Recovery",           f"+{retrain['recovery_pp']}pp")
    t2.add_row("Gap to baseline",    f"{retrain['gap_to_baseline_pp']}pp")
    t2.add_row("Examples removed",   f"{retrain['examples_removed']} ({retrain['filter_pct']}%)")
    t2.add_row("Dominant failure",   retrain['dominant_failure_mode'])
    console.print(t2)

    # Inference
    t3 = Table(title="Inference Stack Assessment", box=box.SIMPLE_HEAVY, show_lines=True)
    t3.add_column("Metric", style="bold")
    t3.add_column("Value", justify="right")
    t3.add_column("Note")
    risk_color = {"LOW": "green", "MEDIUM": "yellow", "HIGH": "red"}[inference["oom_risk"]]
    t3.add_row("KV cache allocated",      f"{inference['kv_cache_allocated_gb']} GB", "")
    t3.add_row("Headroom",                f"{inference['headroom_gb']} GB", "")
    t3.add_row("OOM risk",                f"[{risk_color}]{inference['oom_risk']}[/{risk_color}]", "at peak concurrency")
    t3.add_row("TTFT benchmark p99",      f"{inference['ttft_benchmark_p99_ms']} ms", "warm cache, homogeneous load")
    t3.add_row("TTFT production est.",    f"{inference['ttft_production_low_ms']}–{inference['ttft_production_high_ms']} ms", "1.5–2.5x factor for variance")
    t3.add_row("Throughput",              f"{inference['throughput_tok_s']:,.0f} tok/s", "")
    console.print(t3)


def _plain_report(decomp, trajectory, retrain, inference):
    print("\n=== LLM FINE-TUNING AUDIT — SUMMARY ===\n")
    print("BFCL Regression Decomposition")
    print(f"  Baseline strict:       {decomp['baseline_strict_pct']}%")
    print(f"  Fine-tuned strict:     {decomp['finetuned_strict_pct']}%")
    print(f"  Total regression:      {decomp['total_regression_pp']}pp")
    print(f"    — prefix artifact:   {decomp['prefix_artifact_pp']}pp")
    print(f"    — genuine:           {decomp['genuine_regression_pp']}pp")
    print(f"\nCheckpoint trajectory:   {trajectory}")
    print(f"\nFiltered retrain:")
    print(f"  Recovery:              +{retrain['recovery_pp']}pp")
    print(f"  Gap to baseline:       {retrain['gap_to_baseline_pp']}pp")
    print(f"\nInference:")
    print(f"  OOM risk:              {inference['oom_risk']}")
    print(f"  TTFT prod estimate:    {inference['ttft_production_low_ms']}–{inference['ttft_production_high_ms']} ms")


def main():
    parser = argparse.ArgumentParser(description="LLM Fine-Tuning Audit: Regression Analysis")
    parser.add_argument("--results", type=str, default="results/eval_results_redacted.json",
                        help="Path to the redacted evaluation results JSON")
    args = parser.parse_args()

    with open(args.results) as fh:
        data = json.load(fh)
    models = data["models"]

    base = models["baseline_7b"]["bfcl_strict_pct"] / 100
    ft = models["finetuned_7b_lora_r32"]
    filt = models["finetuned_7b_lora_filtered_1epoch"]

    decomp = decompose_regression(
        strict_score=ft["bfcl_strict_pct"] / 100,
        loose_score=ft["bfcl_loose_pct"] / 100,
        baseline_strict=base,
    )

    trajectory = score_checkpoint_trajectory(
        [{"step": c["step"], "strict_pct": c["bfcl_strict_pct"]} for c in data["checkpoints"]]
    )

    retrain = evaluate_filtered_retrain(
        original_strict=ft["bfcl_strict_pct"] / 100,
        filtered_strict=filt["bfcl_strict_pct"] / 100,
        filtered_loose=filt["bfcl_loose_pct"] / 100,
        baseline_strict=base,
        examples_removed=filt["examples_removed"],
        total_examples=filt["total_examples"],
    )

    inference = assess_inference_stack(InferenceConfig(
        model_name="Qwen2.5-7B-Instruct-LoRA-FP8",
        model_size_gb=ft["fp8_size_gb"],
        vram_total_gb=80.0,
        mem_fraction_static=0.90,
        max_concurrency=32,
        max_tokens=256,
        max_context_len=8192,
        quantization="FP8",
        serving_framework="SGLang",
        ttft_p99_ms=ft["ttft_p99_ms"],
        throughput_tok_s=ft["throughput_tok_s"],
        num_layers=28, num_kv_heads=4, head_dim=128,
    ))

    print_report(decomp, trajectory, retrain, inference)

    # Save structured output
    output = {
        "regression_decomposition": decomp,
        "checkpoint_trajectory":    trajectory,
        "filtered_retrain":         retrain,
        "inference_assessment":     inference,
        "failure_mode_taxonomy":    FAILURE_MODES,
    }
    out_path = Path("results/audit_summary.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
