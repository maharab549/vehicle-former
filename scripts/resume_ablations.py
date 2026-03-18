"""
Resume remaining ablations:
  - Ablation 2 (No World Model): resume from best checkpoint, run 10K more steps
  - Ablation 3 (No LLM Prior): fresh 100K run
"""
import subprocess
import os
import time
from datetime import datetime
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
_venv_python = _project_root / "venv" / "Scripts" / "python.exe"
PYTHON = str(_venv_python)


def run(desc, tag, flags, steps, resume_tag=None):
    print(f"\n{'='*70}")
    print(f"  {desc}")
    print(f"  Tag: {tag}  |  Steps: {steps:,}")
    if resume_tag:
        print(f"  Resume from: {resume_tag}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    cmd = [
        PYTHON, "-u", "train.py",
        "--config", "configs/default.yaml",
        "--steps", str(steps),
        "--tag", tag,
    ] + flags
    if resume_tag:
        cmd += ["--resume", resume_tag]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(_project_root)
    env["HF_HOME"] = r"E:\hf_cache"

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(_project_root), text=True, env=env)
    elapsed = time.time() - t0
    status = "SUCCESS" if result.returncode == 0 else f"FAILED (code {result.returncode})"
    print(f"\n  [{status}] {tag} finished in {elapsed/3600:.1f}h")
    return result.returncode == 0


if __name__ == "__main__":
    # Ablation 2: Resume No World Model — run 10K more from best checkpoint
    run(
        desc="Ablation 2/3: No World Model (finishing remaining steps)",
        tag="ablation_no_wm",
        flags=["--no-wm"],
        steps=10_000,
        resume_tag="best",
    )

    # Ablation 3: No LLM Prior — fresh 100K run
    run(
        desc="Ablation 3/3: No LLM Prior",
        tag="ablation_no_llm",
        flags=["--no-llm"],
        steps=100_000,
    )

    print("\n" + "=" * 70)
    print("  ALL REMAINING ABLATIONS COMPLETE")
    print("=" * 70)
