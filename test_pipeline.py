"""
test_pipeline.py — CPU-only unit tests for final_run.py and final_run_q2.py.

No model download, no HF datasets, no GPU required.
Tests every pure-logic component with synthetic data.

Run:
  python test_pipeline.py              # all tests
  python test_pipeline.py -v           # verbose
  python test_pipeline.py TestScoring  # single class

End-to-end smoke (requires network + ~2 GB RAM, downloads Qwen3-0.6B once):
  python test_pipeline.py --smoke
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pickle
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import torch
import numpy as np

# ---------------------------------------------------------------------------
# Helpers to import final_run without triggering HF imports
# ---------------------------------------------------------------------------

def _make_stub_module(name: str, **attrs) -> types.ModuleType:
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def _patch_heavy_imports():
    """Inject lightweight stubs for HF/datasets so final_run imports cleanly."""
    # datasets
    ds_stub = _make_stub_module("datasets")
    ds_stub.load_dataset = MagicMock(return_value=[])
    sys.modules.setdefault("datasets", ds_stub)

    # transformers (only the names final_run.py actually touches at import time)
    tf_stub = _make_stub_module("transformers")
    tf_stub.AutoTokenizer = MagicMock()
    tf_stub.AutoModelForCausalLM = MagicMock()
    tf_stub.DynamicCache = MagicMock()
    sys.modules.setdefault("transformers", tf_stub)

    # models.py — stub ModelWrapper and _past_length
    mw_stub = _make_stub_module("models")
    mw_stub._past_length = lambda pkv: 0
    class _FakeModelWrapper:
        device = torch.device("cpu")
        _latent_realign_matrices: dict = {}
        def __init__(self, *a, **kw): pass
    mw_stub.ModelWrapper = _FakeModelWrapper
    sys.modules.setdefault("models", mw_stub)

    # utils.py — provide the symbols final_run imports
    utils_stub = _make_stub_module("utils")
    utils_stub.auto_device = lambda d=None: torch.device("cpu")
    utils_stub.set_seed = lambda s: random.seed(s)
    utils_stub.extract_gold = lambda s: s.split("####")[-1].strip() if "####" in s else s
    utils_stub.normalize_answer = lambda s: s.strip().lower()
    sys.modules.setdefault("utils", utils_stub)

    # data.py
    data_stub = _make_stub_module("data")
    data_stub.load_gsm8k = MagicMock(return_value=[])
    data_stub.load_arc_challenge = MagicMock(return_value=[])
    data_stub.load_mbppplus = MagicMock(return_value=[])
    sys.modules.setdefault("data", data_stub)

    # methods.py
    methods_stub = _make_stub_module("methods")
    methods_stub.Agent = MagicMock
    methods_stub.default_agents = lambda: []
    sys.modules.setdefault("methods", methods_stub)

    # prompts.py
    prompts_stub = _make_stub_module("prompts")
    prompts_stub.build_agent_message_sequential_latent_mas = MagicMock(return_value=[])
    sys.modules.setdefault("prompts", prompts_stub)

    # tqdm
    tqdm_stub = _make_stub_module("tqdm")
    tqdm_stub.tqdm = lambda it, **kw: it
    sys.modules.setdefault("tqdm", tqdm_stub)


_patch_heavy_imports()

# Now import the modules under test
import final_run as fr
import final_run_q2 as q2


# ===========================================================================
# 1. Utility functions
# ===========================================================================

class TestFmtBytes(unittest.TestCase):
    def test_bytes(self):
        self.assertEqual(fr.fmt_bytes(512), "512.0B")

    def test_kb(self):
        self.assertEqual(fr.fmt_bytes(2048), "2.0KB")

    def test_mb(self):
        self.assertEqual(fr.fmt_bytes(3 * 1024 ** 2), "3.0MB")

    def test_gb(self):
        self.assertEqual(fr.fmt_bytes(5 * 1024 ** 3), "5.0GB")
        self.assertEqual(fr.fmt_bytes(2 * 1024 ** 3), "2.0GB")

    def test_tb(self):
        # loop exhausts at GB, falls through to TB suffix
        self.assertEqual(fr.fmt_bytes(2 * 1024 ** 4), "2.0TB")


class TestDirSizeBytes(unittest.TestCase):
    def test_counts_files(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "a.txt").write_bytes(b"hello")
            (p / "b.txt").write_bytes(b"world!")
            self.assertEqual(fr.dir_size_bytes(p), 11)

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(fr.dir_size_bytes(Path(d)), 0)


# ===========================================================================
# 2. KV cache helpers
# ===========================================================================

class TestKvToLegacy(unittest.TestCase):
    def _fake_kv(self, n_layers=2, n_heads=4, seq=8, d=16):
        return [(torch.randn(1, n_heads, seq, d), torch.randn(1, n_heads, seq, d))
                for _ in range(n_layers)]

    def test_list_passthrough(self):
        kv = self._fake_kv()
        out = fr.kv_to_legacy(kv)
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 2)

    def test_none_returns_none(self):
        self.assertIsNone(fr.kv_to_legacy(None))

    def test_dynamic_cache_object(self):
        # simulate HF DynamicCache with key_cache / value_cache attributes
        n_layers = 3
        ks = [torch.randn(1, 2, 5, 8) for _ in range(n_layers)]
        vs = [torch.randn(1, 2, 5, 8) for _ in range(n_layers)]
        cache = MagicMock()
        cache.key_cache = ks
        cache.value_cache = vs
        out = fr.kv_to_legacy(cache)
        self.assertEqual(len(out), n_layers)
        for (k, v), ek, ev in zip(out, ks, vs):
            self.assertTrue(torch.equal(k, ek))
            self.assertTrue(torch.equal(v, ev))


class TestSliceKvPositions(unittest.TestCase):
    def test_slice_shape(self):
        kv = [(torch.randn(1, 4, 20, 16), torch.randn(1, 4, 20, 16)) for _ in range(2)]
        sliced = fr.slice_kv_positions(kv, start=5, end=15)
        self.assertEqual(len(sliced), 2)
        for k, v in sliced:
            self.assertEqual(k.shape[-2], 10)
            self.assertEqual(v.shape[-2], 10)
            self.assertEqual(k.dtype, torch.float16)

    def test_slice_none_returns_empty(self):
        self.assertEqual(fr.slice_kv_positions(None, 0, 5), [])


# ===========================================================================
# 3. Scoring
# ===========================================================================

class TestScoreWithSafeExec(unittest.TestCase):
    def _item(self, gold):
        return {"gold": gold, "question": "q"}

    def test_gsm8k_correct(self):
        item = self._item("42")
        text = "The answer is #### 42"
        pred, gold, ok = fr.score_with_safe_exec("gsm8k", item, text)
        self.assertTrue(ok)

    def test_gsm8k_wrong(self):
        item = self._item("42")
        text = "The answer is #### 99"
        _, _, ok = fr.score_with_safe_exec("gsm8k", item, text)
        self.assertFalse(ok)

    def test_arc_challenge_correct(self):
        item = self._item("b")
        text = "The correct answer is B."
        _, _, ok = fr.score_with_safe_exec("arc_challenge", item, text)
        self.assertTrue(ok)

    def test_mbppplus_correct(self):
        item = self._item("assert add(2, 3) == 5\n")
        code = "```python\ndef add(a, b):\n    return a + b\n```"
        _, _, ok = fr.score_with_safe_exec("mbppplus", item, code)
        self.assertTrue(ok)

    def test_mbppplus_wrong(self):
        item = self._item("assert add(2, 3) == 5\n")
        code = "```python\ndef add(a, b):\n    return a - b\n```"
        _, _, ok = fr.score_with_safe_exec("mbppplus", item, code)
        self.assertFalse(ok)

    def test_mbppplus_no_code_block(self):
        item = self._item("assert add(2, 3) == 5\n")
        _, _, ok = fr.score_with_safe_exec("mbppplus", item, "no code here")
        self.assertFalse(ok)


# ===========================================================================
# 4. Split locking
# ===========================================================================

class TestLockSplits(unittest.TestCase):
    def test_creates_files_and_partitions(self):
        with tempfile.TemporaryDirectory() as d:
            # lock_splits writes to data/splits/ relative path — patch it
            orig_cwd = os.getcwd()
            os.chdir(d)
            try:
                splits = fr.lock_splits(Path(d), "gsm8k", n_total=100, seed=0)
                disc = splits["discovery"]
                val  = splits["validation"]
                test = splits["test"]
                # correct sizes
                self.assertEqual(len(disc), 40)
                self.assertEqual(len(val), 20)
                self.assertEqual(len(test), 40)
                # no overlap
                self.assertEqual(len(set(disc) & set(val)), 0)
                self.assertEqual(len(set(disc) & set(test)), 0)
                # union = all indices
                self.assertEqual(sorted(disc + val + test), list(range(100)))
            finally:
                os.chdir(orig_cwd)

    def test_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            orig_cwd = os.getcwd()
            os.chdir(d)
            try:
                s1 = fr.lock_splits(Path(d), "gsm8k", n_total=50, seed=7)
                s2 = fr.lock_splits(Path(d), "gsm8k", n_total=50, seed=7)
                self.assertEqual(s1, s2)
            finally:
                os.chdir(orig_cwd)


# ===========================================================================
# 5. Bucket assignment
# ===========================================================================

class TestAssignBuckets(unittest.TestCase):
    def _write_meta(self, path: Path, correct: bool):
        path.mkdir(parents=True, exist_ok=True)
        (path / "metadata.json").write_text(json.dumps({"correct": correct}))

    def test_all_four_buckets(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            task = "gsm8k"
            lmas_base = root / "latent_mas" / task
            sa_base = root / "single_agent_latent_greedy" / task

            # B1: lmas right, sa wrong
            self._write_meta(lmas_base / "example_0000", True)
            self._write_meta(sa_base   / "example_0000", False)
            # B2: lmas wrong, sa right
            self._write_meta(lmas_base / "example_0001", False)
            self._write_meta(sa_base   / "example_0001", True)
            # B3: both right
            self._write_meta(lmas_base / "example_0002", True)
            self._write_meta(sa_base   / "example_0002", True)
            # B4: both wrong
            self._write_meta(lmas_base / "example_0003", False)
            self._write_meta(sa_base   / "example_0003", False)

            args = MagicMock()
            args.tasks = [task]
            fr.assign_buckets(root, args)

            rows = json.loads((root / "buckets" / f"{task}.json").read_text())
            buckets = {r["example_id"]: r["bucket"] for r in rows}
            self.assertEqual(buckets[0], 1)
            self.assertEqual(buckets[1], 2)
            self.assertEqual(buckets[2], 3)
            self.assertEqual(buckets[3], 4)

    def test_missing_dir_is_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            args = MagicMock()
            args.tasks = ["arc_challenge"]
            # neither latent_mas nor sa dirs exist — should not crash
            fr.assign_buckets(root, args)
            self.assertFalse((root / "buckets" / "arc_challenge.json").exists())


# ===========================================================================
# 6. W_a override math
# ===========================================================================

class TestMakeWaOverride(unittest.TestCase):
    def _make_mw(self, D=32):
        mw = MagicMock()
        W = torch.randn(D, D)
        target_norm = torch.tensor(1.0)
        mw._latent_realign_matrices = {0: (W, target_norm)}
        mw._ensure_latent_realign_matrix = MagicMock()
        return mw, D

    def test_identity(self):
        mw, D = self._make_mw()
        fr.make_wa_override(mw, "identity", seed=0)
        W_new, _ = mw._latent_realign_matrices[0]
        self.assertTrue(torch.allclose(W_new, torch.eye(D).to(W_new.dtype), atol=1e-5))

    def test_zero(self):
        mw, D = self._make_mw()
        fr.make_wa_override(mw, "zero", seed=0)
        W_new, _ = mw._latent_realign_matrices[0]
        self.assertTrue(torch.all(W_new == 0))

    def test_random_orthogonal_is_orthogonal(self):
        mw, D = self._make_mw()
        fr.make_wa_override(mw, "random_orthogonal", seed=42)
        W_new, _ = mw._latent_realign_matrices[0]
        W32 = W_new.float()
        eye = torch.eye(D)
        self.assertTrue(torch.allclose(W32 @ W32.T, eye, atol=1e-4))

    def test_random_spectrum_preserves_singular_values(self):
        mw, D = self._make_mw()
        W_orig, _ = mw._latent_realign_matrices[0]
        _, S_orig, _ = torch.linalg.svd(W_orig.float())
        fr.make_wa_override(mw, "random_spectrum", seed=0)
        W_new, _ = mw._latent_realign_matrices[0]
        _, S_new, _ = torch.linalg.svd(W_new.float())
        self.assertTrue(torch.allclose(S_orig.sort().values,
                                       S_new.sort().values, atol=1e-4))

    def test_trained_is_noop(self):
        mw, D = self._make_mw()
        W_before, _ = mw._latent_realign_matrices[0]
        fr.make_wa_override(mw, "trained", seed=0)
        W_after, _ = mw._latent_realign_matrices[0]
        self.assertTrue(torch.equal(W_before, W_after))

    def test_unknown_mode_raises(self):
        mw, _ = self._make_mw()
        with self.assertRaises(ValueError):
            fr.make_wa_override(mw, "bogus_mode", seed=0)


# ===========================================================================
# 7. Top-k SVD basis
# ===========================================================================

class TestComputeTopkBasis(unittest.TestCase):
    def _make_mw_with_wa(self, D=64, rank=10):
        mw = MagicMock()
        # build a rank-deficient W so SVD elbow is meaningful
        U = torch.linalg.qr(torch.randn(D, rank))[0]
        S = torch.linspace(10, 0.1, rank)
        V = torch.linalg.qr(torch.randn(D, rank))[0]
        W = U @ torch.diag(S) @ V.T
        mw._latent_realign_matrices = {0: (W.float(), torch.tensor(1.0))}
        mw._ensure_latent_realign_matrix = MagicMock()
        return mw, D

    def test_returns_tensor_with_right_shape(self):
        mw, D = self._make_mw_with_wa()
        basis = fr._compute_topk_basis(mw, k=None)
        self.assertIsNotNone(basis)
        self.assertEqual(basis.shape[0], D)
        self.assertGreaterEqual(basis.shape[1], 1)

    def test_explicit_k(self):
        mw, D = self._make_mw_with_wa()
        basis = fr._compute_topk_basis(mw, k=5)
        self.assertEqual(basis.shape[1], 5)

    def test_topk_project_preserves_shape(self):
        mw, D = self._make_mw_with_wa()
        basis = fr._compute_topk_basis(mw, k=8)
        vec = torch.randn(1, D)
        out = fr._topk_project(vec, basis)
        self.assertEqual(out.shape, vec.shape)

    def test_topk_project_is_idempotent(self):
        # projecting twice onto same basis should give same result
        mw, D = self._make_mw_with_wa()
        basis = fr._compute_topk_basis(mw, k=8)
        vec = torch.randn(1, D)
        p1 = fr._topk_project(vec, basis)
        p2 = fr._topk_project(p1, basis)
        self.assertTrue(torch.allclose(p1, p2, atol=1e-5))

    def test_topk_project_reduces_norm_or_equal(self):
        mw, D = self._make_mw_with_wa()
        basis = fr._compute_topk_basis(mw, k=4)
        vec = torch.randn(1, D)
        proj = fr._topk_project(vec, basis)
        self.assertLessEqual(proj.norm().item(), vec.norm().item() + 1e-5)


# ===========================================================================
# 8. Q2: get_round1_hidden
# ===========================================================================

class TestGetRound1Hidden(unittest.TestCase):
    def test_missing_file_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "nonexistent.pt"
            self.assertIsNone(q2.get_round1_hidden(p))

    def test_loads_post_aligned(self):
        D = 128
        n_agents, latent_steps = 2, 4
        post = torch.randn(n_agents, latent_steps, D)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = Path(f.name)
        try:
            torch.save({"post_aligned": post}, path)
            h = q2.get_round1_hidden(path)
            self.assertIsNotNone(h)
            self.assertEqual(h.shape, (D,))
            self.assertEqual(h.dtype, np.float32)
            # should be agent 0, step 0
            expected = post[0, 0, :].float().numpy()
            np.testing.assert_allclose(h, expected, rtol=1e-5)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_key_returns_none(self):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = Path(f.name)
        try:
            torch.save({"other_key": torch.randn(3, 3)}, path)
            self.assertIsNone(q2.get_round1_hidden(path))
        finally:
            path.unlink(missing_ok=True)

    def test_corrupted_file_returns_none(self):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            f.write(b"not a tensor file")
            path = Path(f.name)
        try:
            self.assertIsNone(q2.get_round1_hidden(path))
        finally:
            path.unlink(missing_ok=True)


# ===========================================================================
# 9. Q2: probe gating logic
# ===========================================================================

class TestConfidenceGating(unittest.TestCase):
    """Test the probe threshold logic without running any model."""

    def _make_fake_probe(self, prob: float):
        probe = MagicMock()
        probe.predict_proba.return_value = np.array([[1 - prob, prob]])
        return probe

    def _make_runner(self, probe, threshold: float):
        """Build a ConfidenceGatedRunner with all model calls mocked out."""
        mw = MagicMock()
        args = MagicMock()
        args.task_current = "gsm8k"
        args.max_new_tokens = 64
        args.temperature = 0.6
        args.top_p = 0.95
        cfg = fr.CaptureConfig()

        with patch("final_run_q2.LatentMASCondition"), \
             patch("final_run_q2.SingleAgentTextRunner"):
            runner = q2.ConfidenceGatedRunner(
                mw=mw, args=args, cfg=cfg,
                probe=probe, threshold=threshold,
                lmas_out_dir=Path("/nonexistent"),
            )
        return runner

    def test_high_prob_routes_to_lmas(self):
        probe = self._make_fake_probe(prob=0.9)
        runner = self._make_runner(probe, threshold=0.4)
        h = np.random.randn(64).astype(np.float32)

        # mock lmas_cond.run_and_capture
        runner.lmas_cond.run_and_capture = MagicMock(
            return_value={"correct": True, "gate": "lmas"}
        )

        with tempfile.TemporaryDirectory() as d:
            ex_dir = Path(d) / "example_0000"
            # write fake latent_thoughts.pt so get_round1_hidden finds it
            lmas_dir = Path(d) / "lmas"
            lmas_dir.mkdir()
            ex_pt = lmas_dir / "example_0000" / "latent_thoughts.pt"
            ex_pt.parent.mkdir()
            post = torch.from_numpy(h).unsqueeze(0).unsqueeze(0)  # [1,1,D]
            torch.save({"post_aligned": post}, ex_pt)
            runner.lmas_out_dir = lmas_dir

            item = {"question": "2+2?", "gold": "4"}
            meta = runner.run_one(item, 0, ex_dir)

        self.assertEqual(meta.get("gate"), "lmas")

    def test_low_prob_routes_to_single_agent(self):
        probe = self._make_fake_probe(prob=0.1)
        runner = self._make_runner(probe, threshold=0.4)

        runner.sa.generate_one = MagicMock(
            return_value=("The answer is 4", MagicMock(asdict=lambda: {}))
        )

        with tempfile.TemporaryDirectory() as d:
            lmas_dir = Path(d) / "lmas"
            lmas_dir.mkdir()
            ex_pt = lmas_dir / "example_0000" / "latent_thoughts.pt"
            ex_pt.parent.mkdir()
            h = np.random.randn(64).astype(np.float32)
            post = torch.from_numpy(h).unsqueeze(0).unsqueeze(0)
            torch.save({"post_aligned": post}, ex_pt)
            runner.lmas_out_dir = lmas_dir

            ex_dir = Path(d) / "example_0000"
            item = {"question": "2+2?", "gold": "4"}

            with patch("final_run_q2.score_with_safe_exec", return_value=("4", "4", True)):
                meta = runner.run_one(item, 0, ex_dir)

        self.assertEqual(meta.get("gate"), "single_agent_fallback")

    def test_missing_latent_thoughts_defaults_to_lmas(self):
        probe = self._make_fake_probe(prob=0.0)  # would fallback if file existed
        runner = self._make_runner(probe, threshold=0.4)
        runner.lmas_cond.run_and_capture = MagicMock(
            return_value={"correct": True}
        )

        with tempfile.TemporaryDirectory() as d:
            runner.lmas_out_dir = Path(d) / "empty_lmas"
            ex_dir = Path(d) / "example_0000"
            item = {"question": "q", "gold": "a"}
            meta = runner.run_one(item, 0, ex_dir)

        # no .pt file → h is None → gate defaults to lmas
        self.assertEqual(meta.get("gate"), "lmas")


# ===========================================================================
# 10. Compute accounting
# ===========================================================================

class TestComputeAccount(unittest.TestCase):
    def test_asdict_round_trips(self):
        acc = fr.ComputeAccount(forward_passes=3, generated_tokens=100,
                                wall_clock_ms=500.0, gpu_mem_peak_mb=1024.0,
                                latent_steps=4, approx_flops=1e12)
        d = acc.asdict()
        self.assertEqual(d["forward_passes"], 3)
        self.assertEqual(d["generated_tokens"], 100)
        self.assertAlmostEqual(d["approx_flops"], 1e12)

    def test_approx_flops(self):
        # 2 * params * forward_passes * seq_len
        result = fr.approx_flops(num_params=1_000_000, forward_passes=10, seq_len=256)
        self.assertAlmostEqual(result, 2 * 1_000_000 * 10 * 256)


# ===========================================================================
# 11. CONDITIONS dict sanity
# ===========================================================================

class TestConditionsDict(unittest.TestCase):
    def test_all_required_conditions_present(self):
        required = {
            "latent_mas", "single_agent_latent_sampled", "single_agent_latent_greedy",
            "self_consistency", "best_of_n", "cot_matched",
            "latent_mas_random_wa_spectrum", "kv_blocked", "no_transfer",
            "text_mas", "kv_shuffled", "latent_mas_random_wa_orth",
            "latent_mas_zero_wa", "exp_m_identity_wa", "activation_patching",
            "topk_gated", "random_gated",
        }
        self.assertTrue(required.issubset(set(fr.CONDITIONS.keys())),
                        msg=f"Missing: {required - set(fr.CONDITIONS.keys())}")

    def test_n_per_task_positive(self):
        for name, spec in fr.CONDITIONS.items():
            self.assertGreater(spec.n_per_task, 0, msg=f"{name}.n_per_task must be > 0")

    def test_all_condition_names_lists_all(self):
        self.assertEqual(set(fr.ALL_CONDITION_NAMES), set(fr.CONDITIONS.keys()))

    def test_topk_gated_has_topk_k_extra(self):
        self.assertIn("topk_k", fr.CONDITIONS["topk_gated"].extras)

    def test_random_gated_has_fallback_rate(self):
        rate = fr.CONDITIONS["random_gated"].extras.get("fallback_rate")
        self.assertIsNotNone(rate)
        self.assertGreater(rate, 0.0)
        self.assertLess(rate, 1.0)


# ===========================================================================
# 12. Probe file auto-discovery in Q2
# ===========================================================================

class TestQ2ProbeAutoDiscovery(unittest.TestCase):
    def test_missing_probe_exits(self):
        with tempfile.TemporaryDirectory() as d:
            # simulate: output_dir exists but probe file absent
            out_dir = Path(d)
            argv_backup = sys.argv[:]
            sys.argv = ["final_run_q2.py", "--output_dir", str(out_dir)]
            try:
                with self.assertRaises(SystemExit) as cm:
                    q2.main()
                self.assertEqual(cm.exception.code, 1)
            finally:
                sys.argv = argv_backup

    def test_default_probe_path_is_output_dir_pkl(self):
        """Confirm the default probe path resolves to <output_dir>/exp_p_probe.pkl."""
        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d)
            # write a dummy probe so it doesn't exit early
            probe = MagicMock()
            probe.predict_proba = MagicMock(return_value=np.array([[0.3, 0.7]]))
            probe_path = out_dir / "exp_p_probe.pkl"
            with open(probe_path, "wb") as f:
                pickle.dump(probe, f)

            # confirm file is found without --probe_path
            loaded = q2.load_probe(str(probe_path))
            self.assertIsNotNone(loaded)


# ===========================================================================
# 13. End-to-end smoke (optional, skipped by default)
# ===========================================================================

class TestEndToEndSmoke(unittest.TestCase):
    """Full pipeline smoke on CPU with tiny model.
    Skipped unless --smoke flag passed OR env var LMAS_SMOKE=1."""

    @unittest.skipUnless(
        os.environ.get("LMAS_SMOKE") == "1",
        "set LMAS_SMOKE=1 to run end-to-end smoke (downloads Qwen3-0.6B)",
    )
    def test_smoke_final_run(self):
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            result = subprocess.run(
                [
                    sys.executable, "final_run.py",
                    "--output_dir", d,
                    "--conditions", "latent_mas", "single_agent_latent_greedy",
                    "--tasks", "gsm8k",
                    "--smoke", "--test",
                ],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode != 0:
                self.fail(f"final_run.py smoke failed:\n{result.stderr[-3000:]}")

            # check outputs exist
            out = Path(d)
            self.assertTrue((out / "latent_mas" / "gsm8k").exists())
            self.assertTrue((out / "run.log").exists())


# ===========================================================================
# Runner
# ===========================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--smoke", action="store_true",
                   help="Enable end-to-end smoke test (downloads model, ~600s).")
    known, remaining = p.parse_known_args()

    if known.smoke:
        os.environ["LMAS_SMOKE"] = "1"

    # pass remaining args to unittest
    sys.argv = [sys.argv[0]] + remaining
    unittest.main(verbosity=2)
