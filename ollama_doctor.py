#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ollama_doctor - turn "Ollama won't start" into an ordered, evidence-backed fix list.

Why this exists
---------------
Almost every "Ollama is broken" ticket is one of a dozen repeating causes, and the
server log already says which one. What people lack is an order to try them in,
and a way to tell a real GPU problem from a model that never finished downloading.

Usage
-----
    python ollama_doctor.py                          # auto-collect + auto-detect
    python ollama_doctor.py --log path/to/server.log
    python ollama_doctor.py --json                   # machine readable
    python ollama_doctor.py --report report.md

Exit code is 0 when nothing critical was found, 1 when at least one CRITICAL rule
matched, so it can be dropped straight into a CI or a support script.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass, field, asdict

DEFAULT_PORTS = (11434,)
LOG_CANDIDATES = (
    os.path.expanduser("~/.ollama/logs/server.log"),
    "/usr/local/var/log/ollama.log",
    "/var/log/ollama.log",
)


# --------------------------------------------------------------------------- #
# findings
# --------------------------------------------------------------------------- #

@dataclass
class Finding:
    code: str
    severity: str            # critical | warning | info
    title: str
    evidence: str
    fix: str
    verify: str

    def as_dict(self) -> dict:
        return asdict(self)


SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


# --------------------------------------------------------------------------- #
# rules: (code, severity, [regex], title, fix, verify)
# Order inside this table does not matter; the report sorts by severity, then
# by how specific the rule is (later rules are narrower, so they win ties).
# --------------------------------------------------------------------------- #

RULES = [
    # --- port / duplicate instances -----------------------------------------
    ("PORT_IN_USE", "critical",
     [r"bind: address already in use",
      r"listen tcp .*:11434: bind",
      r"only one instance of ollama"],
     "Another process already owns port 11434",
     "Two `ollama serve` processes (or a leftover from a crashed run) are fighting "
     "for the listener. Find and stop the stale one, then start a single server.",
     "ss -lptn 'sport = :11434'   # then kill the pid that is not your current server"),

    # --- GPU / driver --------------------------------------------------------
    ("CUDA_OOM", "critical",
     [r"cudaMalloc failed",
      r"CUDA error: out of memory",
      r"failed to allocate .*VRAM",
      r"out of memory.*(gpu|vram)"],
     "GPU offload request exceeds available VRAM",
     "The default num_ctx plus the requested layers do not fit. Lower context "
     "first (it is cheap), then cap layers. Do not assume the install is broken.",
     "OLLAMA_NUM_PARALLEL=1 ollama run <model> --verbose   # watch 'offloaded N/M layers'"),

    ("CUDA_MISSING", "critical",
     [r"could not load libcuda",
      r"library not found for -lcuda",
      r"CUDA driver version is insufficient",
      r"failed to initialize CUDA",
      r"no CUDA-capable device"],
     "NVIDIA runtime present but unusable from the Ollama process",
     "Either the driver is older than the CUDA runtime Ollama bundles, or the "
     "process cannot see the device (container without --gpus, or a VM without "
     "passthrough). Compare `nvidia-smi` inside the same shell that runs ollama.",
     "nvidia-smi && ollama run <model> --verbose   # both must see the same device"),

    ("ROCM_MISSING", "warning",
     [r"rocm.*not found", r"hip.*failed", r"no AMD GPU detected"],
     "AMD path unavailable",
     "ROCm libraries are missing or the GPU is not on the supported list; Ollama "
     "will silently fall back to CPU, which looks like 'it starts but is unusable'.",
     "rocm-smi || echo 'no rocm' ; check OLLAMA_LLM_LIBRARY=rocm in env"),

    ("NO_GPU_FALLBACK", "warning",
     [r"no compatible GPUs? (were|was) found",
      r"falling back to (cpu|the cpu)",
      r"cpu only", r"no GPU detected"],
     "Running on CPU only",
     "This is the single most common 'it works but a reply takes minutes' cause. "
     "Not fatal, but it explains the symptom and rules out a corrupt model.",
     "ollama ps   # PROCESSOR column must say GPU, not 100% CPU"),

    # --- model blobs ---------------------------------------------------------
    ("BLOB_CORRUPT", "critical",
     [r"digest mismatch", r"unexpected EOF", r"blob.*not found",
      r"failed to verify", r"sha256.*mismatch", r"pull.*(failed|error)"],
     "Model files are incomplete or corrupt",
     "A half-finished pull leaves a manifest that looks valid but a blob that is "
     "not. Deleting the model and re-pulling is the only reliable repair.",
     "ollama rm <model> && ollama pull <model> && ollama run <model>"),

    # --- disk ---------------------------------------------------------------
    ("DISK_FULL", "critical",
     [r"no space left on device", r"disk quota exceeded", r"ENOSPC"],
     "No free space where blobs live",
     "Models live under ~/.ollama/models (or OLLAMA_MODELS). A full volume fails "
     "mid-pull and then presents as a corrupt model on the next start.",
     "df -h ~/.ollama ; du -sh ~/.ollama/models"),

    # --- memory -------------------------------------------------------------
    ("RAM_SHORT", "warning",
     [r"cannot allocate memory",
      r"oom-kill", r"oom_reaper", r"Out of memory: Killed process",
      r"killed process \d+"],
     "Host ran out of RAM",
     "Context size dominates RAM use far more than parameter count suggests. "
     "Cut num_ctx before buying hardware.",
     "OLLAMA_NUM_PARALLEL=1 ollama run <model> --verbose"),

    # --- permissions --------------------------------------------------------
    ("PERMISSION", "warning",
     [r"permission denied", r"operation not permitted", r"EACCES"],
     "File or device permission problem",
     "systemd units and containers often run as an account that cannot read "
     "~/.ollama, /dev/kvm or the model directory.",
     "id ; ls -ld ~/.ollama ~/.ollama/models"),

    # --- misc ---------------------------------------------------------------
    ("PORT_FOREIGN", "info",
     [r"address already in use by"],
     "A non-Ollama process holds 11434",
     "Some other service grabbed the port before Ollama started; check before "
     "blaming Ollama.",
     "lsof -i :11434"),
]

# Logs are loud: a single failing run can print the same line dozens of times.
# One hit per rule is enough evidence, and the report stays readable.
MAX_EVIDENCE = 220


def scan_text(text: str) -> list[Finding]:
    """Run every rule against the log text. Pure function, unit tested."""
    findings = []
    lines = text.splitlines()
    for code, severity, patterns, title, fix, verify in RULES:
        for pat in patterns:
            rx = re.compile(pat, re.I)
            for ln in lines:
                if rx.search(ln):
                    findings.append(Finding(
                        code=code,
                        severity=severity,
                        title=title,
                        evidence=ln.strip()[:MAX_EVIDENCE],
                        fix=fix,
                        verify=verify,
                    ))
                    break
            else:
                continue
            break
    return findings


# --------------------------------------------------------------------------- #
# environment collection - best effort, never fatal
# --------------------------------------------------------------------------- #

def run(cmd: list[str], timeout: int = 8) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (p.stdout or "").strip() or (p.stderr or "").strip()
    except Exception as exc:                     # missing binary, no perms, ...
        return f"<unavailable: {exc.__class__.__name__}>"


def port_state(port: int) -> str:
    s = socket.socket()
    s.settimeout(1.5)
    try:
        s.connect(("127.0.0.1", port))
        return "open"
    except Exception:
        return "closed"
    finally:
        s.close()


def collect(log_path: str | None) -> dict:
    env: dict = {}
    env["platform"] = f"{platform.system()} {platform.release()} {platform.machine()}"
    env["python"] = platform.python_version()

    env["ollama_version"] = run(["ollama", "--version"])[:80]
    env["cpu"] = run([sys.executable, "-c",
                      "import os;print(os.cpu_count())"])

    # RAM: /proc/meminfo on Linux, GlobalMemoryStatusEx on Windows, sysctl on macOS
    try:
        with open("/proc/meminfo") as fh:
            for ln in fh:
                if ln.startswith("MemTotal"):
                    env["ram_gb"] = round(int(ln.split()[1]) / 1024 / 1024, 1)
                    break
    except Exception:
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                env["ram_gb"] = round(st.ullTotalPhys / 1024 ** 3, 1)
        except Exception:
            env.setdefault("ram_gb", None)

    env["nvidia_smi"] = run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                             "--format=csv,noheader"])[:160]
    env["rocm_smi"] = run(["rocm-smi", "--showproductname"])[:80]

    env["ports"] = {str(p): port_state(p) for p in DEFAULT_PORTS}
    env["models_dir"] = os.environ.get("OLLAMA_MODELS",
                                       os.path.expanduser("~/.ollama/models"))
    if os.path.isdir(env["models_dir"]):
        total, used, free = shutil.disk_usage(env["models_dir"])
        env["disk_free_gb"] = round(free / 1024 ** 3, 1)
        env["disk_note"] = "models dir exists"
    else:
        # Blobs will land here on first pull, so measure the volume that will host
        # them rather than reporting a blank.
        home = os.path.expanduser("~")
        try:
            total, used, free = shutil.disk_usage(home)
            env["disk_free_gb"] = round(free / 1024 ** 3, 1)
        except Exception:
            env["disk_free_gb"] = None
        env["disk_note"] = "models dir missing (measured home volume) - nothing pulled yet"

    if not log_path:
        for cand in LOG_CANDIDATES:
            if os.path.exists(cand):
                log_path = cand
                break
    env["log_path"] = log_path
    return env


def load_log(path: str | None, tail_lines: int = 4000) -> str:
    if not path or not os.path.exists(path):
        return ""
    try:
        with open(path, "r", errors="replace") as fh:
            return "".join(fh.readlines()[-tail_lines:])
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #

def render(env: dict, findings: list[Finding]) -> str:
    findings = sorted(findings, key=lambda f: (SEVERITY_ORDER[f.severity], f.code))
    out = ["# Ollama startup diagnosis", ""]

    out.append("## Environment")
    out.append("")
    for k in ("platform", "ollama_version", "cpu", "ram_gb", "nvidia_smi",
              "disk_free_gb", "disk_note", "log_path"):
        out.append(f"- **{k}**: {env.get(k)}")
    out.append(f"- **port 11434**: {env['ports'].get('11434')}")
    out.append("")

    if not findings:
        out += ["## Findings", "",
                "No known failure signature matched. That usually means the server "
                "starts and the problem is downstream: model not pulled, wrong "
                "context size, or the client pointing at a different port.", ""]
    else:
        out += ["## Findings", ""]
        for i, f in enumerate(findings, 1):
            out.append(f"### {i}. [{f.severity.upper()}] {f.title}  (`{f.code}`)")
            out.append("")
            out.append(f"- Evidence: `{f.evidence}`")
            out.append(f"- What it means: {f.fix}")
            out.append(f"- Verify with: `{f.verify}`")
            out.append("")

    if any(f.severity == "critical" for f in findings):
        out += ["## Do not do this", "",
                "Reinstalling before reading the log destroys the one piece of "
                "evidence that identifies the cause. Peeling the fix in the order "
                "above costs minutes; a blind reinstall costs hours and usually "
                "reproduces the same failure.", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Diagnose Ollama startup failures from evidence.")
    ap.add_argument("--log", help="path to server.log (default: auto-detect)")
    ap.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    ap.add_argument("--report", metavar="PATH", help="write the Markdown report to PATH")
    args = ap.parse_args(argv)

    env = collect(args.log)
    text = load_log(env["log_path"])
    findings = scan_text(text)

    if args.json:
        print(json.dumps({"environment": env, "findings": [f.as_dict() for f in findings]},
                         indent=2, ensure_ascii=False))
    else:
        md = render(env, findings)
        print(md)
        if args.report:
            with open(args.report, "w", encoding="utf-8") as fh:
                fh.write(md)
            print(f"\n[written] {args.report}", file=sys.stderr)

    return 1 if any(f.severity == "critical" for f in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
