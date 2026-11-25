# ollama_doctor

Turn "Ollama won't start" into an ordered, evidence-backed fix list — instead of
guessing, reinstalling, and losing the log that already said what was wrong.

```
$ python ollama_doctor.py --log ~/.ollama/logs/server.log
```

## What it does

1. **Collects facts** — OS, CPU count, RAM, `nvidia-smi` / `rocm-smi` output,
   free disk on the model volume, whether port 11434 is listening, and the log tail.
2. **Matches the log against 10 known failure signatures** — duplicate
   instances on 11434, VRAM shortfall, broken CUDA runtime, ROCm missing,
   silent CPU fallback, corrupt/half-pulled blobs, full disk, host OOM,
   permission errors, foreign processes on the port.
3. **Prints them ranked by severity**, each with the exact log line that
   triggered it, what it means, and the command that confirms the fix.

Exit code `0` = nothing critical found, `1` = at least one critical. Drop it in a
support script or CI without extra glue.

## Why the ordering matters

The most expensive mistake in these tickets is treating "it starts but answers
take 90 seconds" as a broken install. That is almost always silent CPU fallback —
`no compatible GPUs were found` sits in the log while people reinstall three times.
Reading the log first turns a half-day into five minutes.

## Design notes

- **No network calls, no telemetry, no writes** except the optional `--report` file.
- **One finding per rule.** A failing run repeats the same line hundreds of times;
  the report shows the first hit only, capped at 220 characters.
- **CUDA OOM is not host OOM.** `out of memory` inside a `cudaMalloc` line is a
  VRAM problem; host-level rules only fire on `oom-kill` / `cannot allocate memory`.
  There is a regression test for exactly this.
- **Absence is handled.** Missing binary, missing log, missing models dir — each
  degrades to a note, never a crash.
- `scan_text()` is a pure function, which is why the whole rule table is unit-testable
  with plain text fixtures and no running server.

## Tests

```
$ python -m unittest test_doctor -v
Ran 11 tests in 3.5s
OK
```

Fixture logs in `sample_logs/` cover: healthy startup, CPU fallback, VRAM OOM,
port conflict, corrupt blob. `sample_report_cuda_oom.md` is real output, not a mock.

## Usage

```
python ollama_doctor.py                          # auto-detect log + environment
python ollama_doctor.py --log /path/server.log   # explicit log
python ollama_doctor.py --json                   # machine-readable
python ollama_doctor.py --report report.md       # write Markdown
```

Python 3.9+, standard library only.

## Scope

This diagnoses the *server*. It does not fix client-side mismatches (wrong
`OLLAMA_HOST`, proxy interception, browser CORS) — but if the environment block
shows port 11434 `closed` while the model volume is healthy, that is your next
place to look, and the report tells you so by not inventing a failure.
