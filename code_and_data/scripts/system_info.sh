#!/usr/bin/env bash
# Records the execution environment: OS, CPU, cores, RAM, GPU(s), CUDA,
# Python and key library versions. Output goes to stdout and to
# results/aggregated/system_info.txt (each manifest_6X.json additionally
# stores a per-run hardware string).
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/aggregated
OUT="results/aggregated/system_info.txt"

{
    echo "== system_info $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
    echo "-- OS --"
    uname -a
    [[ -f /etc/os-release ]] && grep PRETTY_NAME /etc/os-release
    echo "-- CPU --"
    grep -m1 "model name" /proc/cpuinfo 2>/dev/null || sysctl -n machdep.cpu.brand_string 2>/dev/null
    echo "cores: $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null)"
    echo "-- RAM --"
    grep MemTotal /proc/meminfo 2>/dev/null || echo "n/a"
    echo "-- GPU --"
    if command -v nvidia-smi > /dev/null 2>&1; then
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
        nvidia-smi | grep -m1 "CUDA Version" || true
    else
        echo "no NVIDIA GPU / nvidia-smi not found"
    fi
    echo "-- Python --"
    python3 --version
    python3 - << 'PY'
for mod in ("numpy", "scipy", "tensorflow", "matplotlib"):
    try:
        m = __import__(mod)
        print(f"{mod}=={m.__version__}")
    except Exception as e:
        print(f"{mod}: not importable ({e})")
PY
    echo "-- conda env --"
    echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-<none>}"
} | tee "$OUT"
echo ">>> saved to $OUT"
