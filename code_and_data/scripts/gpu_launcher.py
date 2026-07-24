#!/usr/bin/env python3
"""Dynamic GPU scheduler for the Example 6.3 training grid.

The machine has several GPUs that may be SHARED with other users, so GPUs
are assigned dynamically: before launching a job the scheduler polls
`nvidia-smi` and only uses cards with at least --min-free-mb of free
memory (default 8000 MB, i.e. an almost-free 10 GB card) and no job of
ours already running on them.  Each job runs with

    CUDA_VISIBLE_DEVICES=<idx>  TF_FORCE_GPU_ALLOW_GROWTH=true

so TensorFlow only takes the memory it needs instead of grabbing the
whole card.  The grid itself is defined in configs/exp63.json and printed
by `run_experiment_63.py --print-jobs`; finished runs (run.json with
status "done") are skipped, so the scheduler is fully RESUMABLE -- it can
be killed and restarted at any time.

Usage:
    python3 scripts/gpu_launcher.py                # full grid
    python3 scripts/gpu_launcher.py --smoke        # smoke grid
    python3 scripts/gpu_launcher.py --min-free-mb 6000 --max-parallel 2
    python3 scripts/gpu_launcher.py --dry-run      # show the job list

Without any usable GPU (no nvidia-smi) the jobs run SEQUENTIALLY on CPU
with a warning (useful for --smoke pipeline tests).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXP_SCRIPT = os.path.join(ROOT, "experiments", "run_experiment_63.py")
RUNS_ROOT = os.path.join(ROOT, "results", "raw", "exp63", "runs")
POLL_SECONDS = 30
RETRIES = 1


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def get_jobs(smoke):
    cmd = [sys.executable, EXP_SCRIPT, "--stage", "train", "--print-jobs"]
    if smoke:
        cmd.append("--smoke")
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def job_tag(j):
    return f"lam{j['lam']}_s{j['net_seed']}_{j['sampling']}"


def is_done(j):
    p = os.path.join(RUNS_ROOT, job_tag(j), "run.json")
    if not os.path.exists(p):
        return False
    try:
        with open(p) as f:
            return json.load(f).get("status") == "done"
    except (json.JSONDecodeError, OSError):
        return False


def gpu_free_mb():
    """{gpu_index: free MiB} or None when nvidia-smi is unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return None
        free = {}
        for line in out.stdout.strip().splitlines():
            idx, mb = line.split(",")
            free[int(idx)] = int(mb)
        return free or None
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None


def launch(j, gpu_idx, smoke):
    tag = job_tag(j)
    rdir = os.path.join(RUNS_ROOT, tag)
    os.makedirs(rdir, exist_ok=True)
    env = dict(os.environ)
    env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    if gpu_idx is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    cmd = [sys.executable, EXP_SCRIPT, "--stage", "train",
           "--lam", str(j["lam"]), "--net-seed", str(j["net_seed"]),
           "--sampling", j["sampling"]]
    if smoke:
        cmd.append("--smoke")
    logf = open(os.path.join(rdir, "train.log"), "a")
    logf.write(f"\n===== launch {datetime.now().isoformat()} "
               f"GPU={gpu_idx} =====\n")
    logf.flush()
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=logf, cwd=ROOT)
    where = f"GPU {gpu_idx}" if gpu_idx is not None else "CPU"
    log(f"START {tag} on {where} (pid {proc.pid}, log: "
        f"{os.path.relpath(logf.name, ROOT)})")
    return {"job": j, "tag": tag, "proc": proc, "gpu": gpu_idx,
            "logf": logf, "tries": 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--min-free-mb", type=int, default=8000,
                    help="minimum free GPU memory to consider a card usable "
                         "(the machine may be shared; default 8000)")
    ap.add_argument("--max-parallel", type=int, default=0,
                    help="cap on concurrent jobs (0 = number of usable GPUs)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs = get_jobs(args.smoke)
    pending = [j for j in jobs if not is_done(j)]
    done_already = len(jobs) - len(pending)
    log(f"grid: {len(jobs)} jobs, {done_already} already done, "
        f"{len(pending)} to run")
    for j in pending:
        log(f"  pending: {job_tag(j)}")
    if args.dry_run or not pending:
        return 0

    have_gpu = gpu_free_mb() is not None
    if not have_gpu:
        log("WARNING: nvidia-smi not available -- running jobs "
            "SEQUENTIALLY on CPU.")
        for j in pending:
            a = launch(j, None, args.smoke)
            a["proc"].wait()
            a["logf"].close()
            status = "ok" if is_done(j) else "FAILED"
            log(f"END   {a['tag']}: {status}")
        failed = [job_tag(j) for j in pending if not is_done(j)]
        if failed:
            log(f"FAILED jobs: {failed}")
            return 1
        return 0

    active = []
    failed = []
    while pending or active:
        # reap finished processes
        still = []
        for a in active:
            rc = a["proc"].poll()
            if rc is None:
                still.append(a)
                continue
            a["logf"].close()
            if is_done(a["job"]):
                log(f"END   {a['tag']}: ok (rc={rc})")
            elif a["tries"] <= RETRIES:
                log(f"RETRY {a['tag']} (rc={rc}, attempt "
                    f"{a['tries'] + 1}/{RETRIES + 1})")
                a2 = launch(a["job"], a["gpu"], args.smoke)
                a2["tries"] = a["tries"] + 1
                still.append(a2)
            else:
                log(f"FAIL  {a['tag']} (rc={rc}) -- giving up, see log")
                failed.append(a["tag"])
        active = still

        # try to launch on free cards
        if pending:
            free = gpu_free_mb() or {}
            busy = {a["gpu"] for a in active}
            cap = args.max_parallel or len(free)
            for idx, mb in sorted(free.items(), key=lambda kv: -kv[1]):
                if not pending or len(active) >= cap:
                    break
                if idx in busy or mb < args.min_free_mb:
                    continue
                active.append(launch(pending.pop(0), idx, args.smoke))
                busy.add(idx)
        if pending and not active:
            log(f"no GPU with >= {args.min_free_mb} MB free "
                f"({len(pending)} jobs waiting) -- polling ...")
        time.sleep(POLL_SECONDS if (pending or active) else 0)

    if failed:
        log(f"FAILED jobs: {failed}")
        return 1
    log("all jobs finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
