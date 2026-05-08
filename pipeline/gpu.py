"""GPU resource management — VRAM detection, Ollama awareness, batch size recommendation.

RTX 5060 Ti (16GB VRAM, sm_120, CUDA 13.2) + Ryzen 7 5700G (8C/16T, 32GB RAM)
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass


@dataclass
class GPUStatus:
    available: bool
    name: str
    total_vram_gb: float
    used_vram_gb: float
    free_vram_gb: float
    cuda_version: str
    ollama_active: bool
    ollama_model: str | None


# ── VRAM budget per model (GB, safe upper bound for encoding) ───────────────
_MODEL_VRAM = {
    "BAAI/bge-m3":                       {"base": 2.5, "per_batch": 0.18},  # 1024d, fp16
    "jhgan/ko-sroberta-multitask":        {"base": 0.9, "per_batch": 0.06},  # 768d, fp32
    "sentence-transformers/all-MiniLM-L6-v2": {"base": 0.2, "per_batch": 0.02},
}
_DEFAULT_MODEL_VRAM = {"base": 2.0, "per_batch": 0.15}


def get_gpu_status() -> GPUStatus:
    """Query GPU status via nvidia-smi (no torch dependency)."""
    try:
        # --query-gpu: total, free, used memory (MiB)
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.used,memory.free",
             "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr)

        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        name, total, used, free = lines[0].split(", ")

        # CUDA version from nvidia-smi header
        smi_ver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        cuda_ver = _get_cuda_version()

        total_gb = int(total) / 1024
        used_gb  = int(used) / 1024
        free_gb  = int(free) / 1024

        ollama_active, ollama_model = _check_ollama()

        return GPUStatus(
            available=True,
            name=name.strip(),
            total_vram_gb=round(total_gb, 2),
            used_vram_gb=round(used_gb, 2),
            free_vram_gb=round(free_gb, 2),
            cuda_version=cuda_ver,
            ollama_active=ollama_active,
            ollama_model=ollama_model,
        )

    except (FileNotFoundError, RuntimeError, Exception):
        return GPUStatus(
            available=False, name="CPU", total_vram_gb=0, used_vram_gb=0,
            free_vram_gb=0, cuda_version="N/A", ollama_active=False, ollama_model=None,
        )


def _get_cuda_version() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _check_ollama() -> tuple[bool, str | None]:
    """Check if Ollama is running a model that consumes VRAM."""
    try:
        result = subprocess.run(
            ["ollama", "ps"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return False, None

        lines = result.stdout.strip().splitlines()
        if len(lines) <= 1:  # header only
            return False, None

        # Parse first model line: NAME, ID, SIZE, PROCESSOR, CONTEXT, UNTIL
        parts = lines[1].split()
        if not parts:
            return False, None

        model_name = parts[0]
        # Check if GPU is involved (PROCESSOR column contains "GPU")
        processor_info = " ".join(parts[3:6]) if len(parts) > 3 else ""
        uses_gpu = "GPU" in processor_info or "gpu" in processor_info.lower()
        return uses_gpu, model_name if uses_gpu else None

    except (FileNotFoundError, Exception):
        return False, None


def recommend_batch_size(model_name: str, free_vram_gb: float, safety_factor: float = 0.75) -> int:
    """Calculate safe batch size given available VRAM.

    Args:
        model_name    : embedding model HuggingFace ID
        free_vram_gb  : available VRAM in GB
        safety_factor : fraction of free VRAM to use (default 0.75)

    Returns:
        recommended batch_size (1–64)
    """
    budget = free_vram_gb * safety_factor
    mvram  = _MODEL_VRAM.get(model_name, _DEFAULT_MODEL_VRAM)
    usable = budget - mvram["base"]

    if usable <= 0:
        return 1  # barely enough — single item

    batch = max(1, int(usable / mvram["per_batch"]))
    return min(batch, 64)  # cap at 64


def print_gpu_summary(status: GPUStatus) -> None:
    """Print a formatted GPU status summary."""
    print("=" * 55)
    print("[GPU] System Resource Check")
    print("=" * 55)
    if not status.available:
        print("  GPU: Not available — CPU mode")
        print("=" * 55)
        return

    bar_used = int(status.used_vram_gb / status.total_vram_gb * 20)
    bar = "█" * bar_used + "░" * (20 - bar_used)

    print(f"  GPU:   {status.name}")
    print(f"  VRAM:  [{bar}] {status.used_vram_gb:.1f}/{status.total_vram_gb:.1f} GB used")
    print(f"  Free:  {status.free_vram_gb:.1f} GB")

    if status.ollama_active and status.ollama_model:
        print(f"\n  ⚠  Ollama [{status.ollama_model}] is using GPU VRAM.")
        print( "     Free VRAM will increase after Ollama unloads (idle ~2 min).")
        print( "     To free now: ollama stop <model>  or  ollama ps")

    print("=" * 55)


def auto_batch_size(model_name: str, requested: int, verbose: bool = True) -> tuple[int, str]:
    """Return the effective batch_size and device, auto-adjusted for available VRAM.

    Returns: (batch_size, device)
    """
    status = get_gpu_status()

    if not status.available:
        if verbose:
            print("[GPU] No CUDA GPU — using CPU")
        return requested, "cpu"

    if verbose:
        print_gpu_summary(status)

    safe = recommend_batch_size(model_name, status.free_vram_gb)

    if safe < requested:
        if verbose:
            print(f"[GPU] Reducing batch_size: {requested} → {safe} "
                  f"(free VRAM {status.free_vram_gb:.1f} GB)")
        return safe, "cuda"

    if verbose:
        print(f"[GPU] batch_size={requested} OK (free VRAM {status.free_vram_gb:.1f} GB)")
    return requested, "cuda"


def optimal_workers(max_workers: int | None = None) -> int:
    """Return optimal number of multiprocessing workers for parameter tuning.

    Strategy for Ryzen 7 5700G (8C/16T, 32GB RAM):
      - Reserve 2 logical cores for OS + Ollama
      - Tuning workers each use ~2-3GB RAM
      - Hard cap at physical core count
    """
    import multiprocessing
    physical = multiprocessing.cpu_count() // 2  # logical → physical estimate
    # Each worker needs ~2.5GB RAM for embeddings + UMAP; cap at 32GB / 2.5 ≈ 12
    ram_cap = 10  # conservative for 32GB
    safe = min(physical - 1, ram_cap, 8)
    safe = max(safe, 1)
    if max_workers:
        safe = min(safe, max_workers)
    return safe


# ── cuML (RAPIDS) availability ────────────────────────────────────────────────

def has_cuml() -> bool:
    """Return True if RAPIDS cuML is importable (WSL2 + RAPIDS 25.04+ / sm_120)."""
    try:
        import cuml  # type: ignore[import]  # noqa: F401  # WSL2-only package
        return True
    except ImportError:
        return False


def make_umap(
    n_neighbors: int = 15,
    n_components: int = 10,
    min_dist: float = 0.0,
    metric: str = "cosine",
    random_state: int = 42,
    use_cuml: bool | None = None,
):
    """Return UMAP model — cuML GPU when available, umap-learn CPU otherwise.

    Args:
        use_cuml: True=force cuML, False=force CPU, None=auto-detect.
    """
    _cuml = has_cuml() if use_cuml is None else use_cuml
    if _cuml:
        from cuml.manifold import UMAP as cuUMAP  # type: ignore[import]
        return cuUMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            output_type="numpy",
        )
    from umap import UMAP  # type: ignore[import]
    return UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        n_jobs=1,
    )


def make_hdbscan(
    min_cluster_size: int = 150,
    min_samples: int = 15,
    cluster_selection_method: str = "eom",
    prediction_data: bool = True,
    use_cuml: bool | None = None,
):
    """Return HDBSCAN model — cuML GPU when available, hdbscan CPU otherwise.

    Args:
        use_cuml: True=force cuML, False=force CPU, None=auto-detect.
    Note:
        cuML HDBSCAN has no ``metric`` param — always uses Euclidean internally.
    """
    _cuml = has_cuml() if use_cuml is None else use_cuml
    if _cuml:
        from cuml.cluster import HDBSCAN as cuHDBSCAN  # type: ignore[import]
        return cuHDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=cluster_selection_method,
            prediction_data=prediction_data,
        )
    from hdbscan import HDBSCAN  # type: ignore[import]
    return HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        metric="euclidean",
        prediction_data=prediction_data,
    )
