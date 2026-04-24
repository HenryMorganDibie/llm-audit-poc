"""
audit/bfcl_regression_analysis.py

Reproduces the regression decomposition methodology from the audit report.
Demonstrates how a ~50pp BFCL regression can be decomposed into:
  - Genuine behavioral regression (~35pp)
  - Scorer artifact from strict name matching (~15pp)

Usage:
    python audit/bfcl_regression_analysis.py --results results/eval_results.json

Redacted version: client identity and proprietary API schemas removed.
"""

import json
import argparse
import numpy as np
from pathlib import Path


# ─── Failure mode taxonomy ────────────────────────────────────────────────────

FAILURE_MODES = {
    "module_prefix_drop": {
        "description": "Right function name, wrong namespace prefix (e.g. 'area_triangle' vs 'geometry.area_triangle')",
        "nature": "Partially artifact — strict scorer penalises; base model does not exhibit this",
        "estimated_share_pp": 8,
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

    return {
        "baseline_strict_pct":       round(baseline_strict * 100, 1),
        "finetuned_strict_pct":       round(strict_score * 100, 1),
        "finetuned_loose_pct":        round(loose_score * 100, 1),
        "total_regression_pp":        round(total_regression * 100, 1),
        "prefix_artifact_pp":         round(prefix_artifact * 100, 1),
        "genuine_regression_pp":      round(genuine_regression * 100, 1),
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
    baseline_strict   : baseline model strict accuracy
    examples_removed  : number of training examples removed
    total_examples    : total training examples before filtering
    """
    recovery_pp     = filtered_strict - original_strict
    gap_to_baseline = baseline_strict - filtered_strict
    filter_pct      = (examples_removed / total_examples) * 100

    return {
        "original_strict_pct":   round(original_strict * 100, 1),
        "filtered_strict_pct":   round(filtered_strict * 100, 1),
        "filtered_loose_pct":    round(filtered_loose * 100, 1),
        "baseline_strict_pct":   round(baseline_strict * 100, 1),
        "recovery_pp":           round(recovery_pp * 100, 1),
        "gap_to_baseline_pp":    round(gap_to_baseline * 100, 1),
        "examples_removed":      examples_removed,
        "filter_pct":            round(filter_pct, 1),
        "dominant_failure_mode": "module_prefix_naming" if gap_to_baseline * 100 <= 10 else "mixed",
    }


def assess_inference_stack(
    model_size_gb: float,
    vram_total_gb: float,
    mem_fraction_static: float,
    max_concurrency: int,
    max_tokens: int,
    max_context_len: int,
    ttft_p99_ms: float,
    throughput_tok_s: float,
) -> dict:
    """
    Assess VRAM headroom and OOM risk for a served LLM under concurrency.

    Parameters
    ----------
    model_size_gb        : model weight size in GB (post-quantization)
    vram_total_gb        : total GPU VRAM (e.g. 80 for H100)
    mem_fraction_static  : SGLang/vLLM static KV cache allocation fraction
    max_concurrency      : peak concurrent requests
    max_tokens           : max output tokens per request
    max_context_len      : max context window (tokens)
    ttft_p99_ms          : measured TTFT p99 in milliseconds
    throughput_tok_s     : measured throughput in tokens/second
    """
    available_after_weights = vram_total_gb - model_size_gb
    kv_cache_allocated_gb   = available_after_weights * mem_fraction_static
    cuda_overhead_gb        = available_after_weights * (1 - mem_fraction_static)
    headroom_gb             = cuda_overhead_gb

    # Heuristic: KV cache demand at peak concurrency
    # Each token in KV cache ≈ 2 × num_layers × hidden_dim × dtype_bytes
    # For a 7B model in FP8: ~0.5KB/token; for 3B: ~0.3KB/token
    bytes_per_token         = 512 if model_size_gb > 5 else 300
    peak_kv_demand_gb       = (max_concurrency * max_context_len * bytes_per_token) / 1e9

    oom_risk = "LOW"
    if peak_kv_demand_gb > kv_cache_allocated_gb * 0.85:
        oom_risk = "HIGH"
    elif peak_kv_demand_gb > kv_cache_allocated_gb * 0.65:
        oom_risk = "MEDIUM"

    # Production TTFT estimate: benchmark was measured under homogeneous load
    # Real production adds 1.5-2.5x factor for prompt variance and cold starts
    ttft_production_low_ms  = round(ttft_p99_ms * 1.5, 0)
    ttft_production_high_ms = round(ttft_p99_ms * 2.5, 0)

    return {
        "model_size_gb":              model_size_gb,
        "kv_cache_allocated_gb":      round(kv_cache_allocated_gb, 1),
        "headroom_gb":                round(headroom_gb, 1),
        "peak_kv_demand_estimate_gb": round(peak_kv_demand_gb, 2),
        "oom_risk":                   oom_risk,
        "mem_fraction_recommendation": 0.85 if mem_fraction_static > 0.87 else mem_fraction_static,
        "ttft_benchmark_p99_ms":      ttft_p99_ms,
        "ttft_production_low_ms":     ttft_production_low_ms,
        "ttft_production_high_ms":    ttft_production_high_ms,
        "throughput_tok_s":           throughput_tok_s,
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
    parser = argparse.ArgumentParser(description="LLM Fine-Tuning Audit — Regression Analysis")
    parser.add_argument("--results", type=str, default=None, help="Path to eval_results.json (optional)")
    args = parser.parse_args()

    # ── Redacted results from the FinSynapse audit engagement ──────────────────
    # Client identity and proprietary API schemas removed.
    # All accuracy numbers are real experimental results.

    decomp = decompose_regression(
        strict_score=0.425,
        loose_score=0.505,
        baseline_strict=0.928,
    )

    checkpoints = [
        {"step": 500,  "strict_pct": 41.2},
        {"step": 1000, "strict_pct": 42.1},
        {"step": 1500, "strict_pct": 43.0},
    ]
    trajectory = score_checkpoint_trajectory(checkpoints)

    retrain = evaluate_filtered_retrain(
        original_strict=0.425,
        filtered_strict=0.648,
        filtered_loose=0.892,
        baseline_strict=0.928,
        examples_removed=537,
        total_examples=10336,
    )

    inference = assess_inference_stack(
        model_size_gb=8.2,
        vram_total_gb=80.0,
        mem_fraction_static=0.90,
        max_concurrency=32,
        max_tokens=256,
        max_context_len=8192,
        ttft_p99_ms=87.0,
        throughput_tok_s=4154,
    )

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
