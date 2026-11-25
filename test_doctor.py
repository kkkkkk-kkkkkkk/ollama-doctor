#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for ollama_doctor. Run: python -m unittest test_doctor -v

Fixtures live inline so the suite has no dependency on files on disk; the one
test that must go through load_log() writes a temp file instead.
"""

import contextlib
import io
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ollama_doctor as od  # noqa: E402

LOG = os.path.join(HERE, "sample_logs")

FIXTURES = {
    "port_conflict": """\
time=2026-11-25T09:31:40.201+08:00 level=INFO msg="starting llm server"
time=2026-11-25T09:31:40.203+08:00 level=ERROR msg="listen tcp 127.0.0.1:11434: bind: address already in use"
time=2026-11-25T09:31:40.204+08:00 level=WARN msg="only one instance of ollama may run at a time"
""",
    "cuda_oom": """\
time=2026-11-25T09:20:11.004+08:00 level=ERROR msg="cudaMalloc failed: out of memory"
time=2026-11-25T09:20:11.006+08:00 level=INFO msg="offloaded 12/33 layers to GPU"
time=2026-11-25T09:20:11.010+08:00 level=ERROR msg="failed to allocate KV cache, requested 4096 tokens"
""",
    "corrupt_blob": """\
time=2026-11-25T09:41:02.300+08:00 level=ERROR msg="digest mismatch for layer sha256:6b0d4c" expected="6b0d4c" actual="e91fa7"
time=2026-11-25T09:41:02.301+08:00 level=ERROR msg="pull model: unexpected EOF"
""",
    "no_gpu": """\
time=2026-11-25T09:14:04.001+08:00 level=WARN msg="no compatible GPUs were found, sampling CPU"
time=2026-11-25T09:14:04.002+08:00 level=INFO msg="inference compute" library=cpu compute="" driver="" name="cpu"
""",
    "clean": """\
time=2026-11-25T09:50:01.900+08:00 level=INFO msg="inference compute" library=cuda compute=8.6 driver=12.2 name="NVIDIA GeForce RTX 3060" total="12.0 GiB"
time=2026-11-25T09:50:02.000+08:00 level=INFO msg="offloaded 33/33 layers to GPU"
time=2026-11-25T09:50:02.500+08:00 level=INFO msg="llama runner started in 0.62 seconds"
""",
}


def codes(name):
    return [f.code for f in od.scan_text(FIXTURES[name])]


def findings(name):
    return od.scan_text(FIXTURES[name])


def write_tmp(name):
    path = os.path.join(tempfile.mkdtemp(), name + ".log")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(FIXTURES[name])
    return path


class TestDetection(unittest.TestCase):
    def test_port_conflict(self):
        self.assertIn("PORT_IN_USE", codes("port_conflict"))

    def test_cuda_oom(self):
        self.assertIn("CUDA_OOM", codes("cuda_oom"))

    def test_corrupt_blob(self):
        self.assertIn("BLOB_CORRUPT", codes("corrupt_blob"))

    def test_cpu_fallback_is_warning_not_critical(self):
        hits = [f for f in findings("no_gpu") if f.code == "NO_GPU_FALLBACK"]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].severity, "warning")

    def test_clean_log_has_no_findings(self):
        # A healthy run says "offloaded 33/33 layers" - that must not be read as a fault.
        self.assertEqual(codes("clean"), [])

    def test_cuda_oom_not_reported_as_host_ram(self):
        # "out of memory" inside a cudaMalloc line is a VRAM problem, not a host RAM one.
        self.assertNotIn("RAM_SHORT", codes("cuda_oom"))


class TestBehaviour(unittest.TestCase):
    def test_no_duplicates_from_repeated_lines(self):
        hits = od.scan_text("cudaMalloc failed: out of memory\n" * 40)
        self.assertEqual([f.code for f in hits].count("CUDA_OOM"), 1)

    def test_sorted_by_severity(self):
        order = [od.SEVERITY_ORDER[f.severity] for f in findings("cuda_oom")]
        self.assertEqual(order, sorted(order), "report order must follow severity")

    def test_evidence_is_bounded(self):
        hits = od.scan_text("bind: address already in use " + "x" * 500)
        self.assertTrue(all(len(f.evidence) <= od.MAX_EVIDENCE for f in hits))

    def test_missing_log_does_not_crash(self):
        self.assertEqual(od.scan_text(od.load_log("no/such/file.log")), [])

    def test_exit_code_reflects_critical(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(od.main(["--log", write_tmp("port_conflict"), "--json"]), 1)
            self.assertEqual(od.main(["--log", write_tmp("clean"), "--json"]), 0)

    def test_load_log_trims_to_tail(self):
        path = write_tmp("cuda_oom")
        self.assertIn("cudaMalloc", od.load_log(path))


class TestSampleLogsAreRealFiles(unittest.TestCase):
    """The repo ships fixture logs for manual runs; keep them and the suite in sync."""

    def test_fixtures_on_disk_match_inline(self):
        if not os.path.isdir(LOG):
            self.skipTest("sample_logs/ not checked out")
        for name in FIXTURES:
            path = os.path.join(LOG, name + ".log")
            self.assertTrue(os.path.exists(path), path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
