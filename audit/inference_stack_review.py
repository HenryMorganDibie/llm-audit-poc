"""
audit/inference_stack_review.py

Evaluates VRAM headroom, KV cache risk, and latency measurement validity
for a served LLM under concurrent load on a single H100.

Redacted version: client identity removed. Numbers are real.
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class InferenceConfig:
    model_name: str
    model_size_gb: float
    vram_total_gb: float
    mem_fraction_static: float
    max_concurrency: int
    max_tokens: int
    max_context_len: int
    quantization: str
    serving_framework: str
    ttft_p99_ms: float
    throughput_tok_s: float


def vram_budget(cfg: InferenceConfig) -> dict:
    available = cfg.vram_total_gb - cfg.model_size_gb
    kv_alloc  = available * cfg.mem_fraction_static
    headroom  = available * (1 - cfg.mem_fraction_static)
    bytes_per_token = 512 if cfg.model_size_gb > 5 else 300
    peak_kv   = (cfg.max_concurrency * cfg.max_context_len * bytes_per_token) / 1e9

    if peak_kv > kv_alloc * 0.85:
        risk = "HIGH"
    elif peak_kv > kv_alloc * 0.65:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "available_after_weights_gb": round(available, 1),
        "kv_cache_allocated_gb":      round(kv_alloc, 1),
        "headroom_gb":                round(headroom, 1),
        "peak_kv_demand_estimate_gb": round(peak_kv, 2),
        "oom_risk":                   risk,
        "recommended_mem_fraction":   0.85 if cfg.mem_fraction_static > 0.87 else cfg.mem_fraction_static,
    }


def latency_validity(cfg: InferenceConfig) -> dict:
    """
    Flag conditions that make benchmark TTFT optimistic vs production reality.
    """
    flags = []
    if cfg.max_tokens <= 256:
        flags.append("Fixed max_tokens — low output variance understates TPOT p99")
    flags.append("Homogeneous prompts — RadixAttention reuse rate will be lower in production")
    flags.append("Warm-cache measurement only — cold start TTFT uncharacterised")

    production_low  = round(cfg.ttft_p99_ms * 1.5)
    production_high = round(cfg.ttft_p99_ms * 2.5)

    return {
        "benchmark_ttft_p99_ms":      cfg.ttft_p99_ms,
        "production_estimate_low_ms": production_low,
        "production_estimate_high_ms":production_high,
        "optimism_flags":             flags,
        "recommendation":             f"Present {production_low}–{production_high}ms as production estimate; cite {cfg.ttft_p99_ms}ms as benchmark floor.",
    }


def run_review(configs: list[InferenceConfig]) -> None:
    results = []
    for cfg in configs:
        vram   = vram_budget(cfg)
        latency= latency_validity(cfg)
        results.append({
            "model":   cfg.model_name,
            "config":  asdict(cfg),
            "vram":    vram,
            "latency": latency,
        })
        print(f"\n{'─'*55}")
        print(f"Model: {cfg.model_name}  ({cfg.quantization}, {cfg.serving_framework})")
        print(f"  VRAM headroom:    {vram['headroom_gb']} GB  |  OOM risk: {vram['oom_risk']}")
        print(f"  KV cache alloc:   {vram['kv_cache_allocated_gb']} GB  |  Peak demand est: {vram['peak_kv_demand_estimate_gb']} GB")
        print(f"  TTFT benchmark:   {latency['benchmark_ttft_p99_ms']} ms p99")
        print(f"  TTFT production:  {latency['production_estimate_low_ms']}–{latency['production_estimate_high_ms']} ms (estimated)")
        for flag in latency["optimism_flags"]:
            print(f"  ⚠  {flag}")

    out = Path("results/inference_review.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    configs = [
        InferenceConfig(
            model_name="Qwen2.5-7B-Instruct-LoRA-FP8",
            model_size_gb=8.2,
            vram_total_gb=80.0,
            mem_fraction_static=0.90,
            max_concurrency=32,
            max_tokens=256,
            max_context_len=8192,
            quantization="FP8",
            serving_framework="SGLang",
            ttft_p99_ms=87.0,
            throughput_tok_s=4154,
        ),
        InferenceConfig(
            model_name="Qwen2.5-3B-Instruct-Baseline-FP16",
            model_size_gb=6.5,
            vram_total_gb=80.0,
            mem_fraction_static=0.90,
            max_concurrency=32,
            max_tokens=256,
            max_context_len=8192,
            quantization="FP16",
            serving_framework="SGLang",
            ttft_p99_ms=78.7,
            throughput_tok_s=6075,
        ),
        InferenceConfig(
            model_name="Qwen2.5-3B-Instruct-Baseline-FP8",
            model_size_gb=3.4,
            vram_total_gb=80.0,
            mem_fraction_static=0.90,
            max_concurrency=32,
            max_tokens=256,
            max_context_len=8192,
            quantization="FP8",
            serving_framework="SGLang",
            ttft_p99_ms=142.9,
            throughput_tok_s=6780,
        ),
    ]
    run_review(configs)
