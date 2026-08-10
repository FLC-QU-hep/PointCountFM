"""Unit tests for the 2D bias-correction apply path.

Covers:
  - F1: bias_mode assertion (1D npz / wrong tag → ValueError)
  - layer-only equivalence (2D == 1D when all bins identical)
  - fallback layer mask path (cells using 1D fallback)
  - R3: out-of-range energy clamps to first/last bin and emits warning

Run with:  PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py"
"""

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from generate_pcfm_cond import apply_bias_correction_2d  # noqa: E402


def _save_2d_npz(
    path,
    bias_factor_2d,
    e_bin_edges,
    fallback=None,
    n_events_per_cell=None,
    bias_factor_1d=None,
    active_layer_mask=None,
    bias_mode="2d",
):
    nl, nbins = bias_factor_2d.shape
    if fallback is None:
        fallback = np.zeros((nl, nbins), dtype=bool)
    if n_events_per_cell is None:
        n_events_per_cell = np.full((nl, nbins), 1000, dtype=np.int64)
    if bias_factor_1d is None:
        bias_factor_1d = bias_factor_2d.mean(axis=1)
    if active_layer_mask is None:
        active_layer_mask = np.ones(nl, dtype=bool)
    np.savez(
        path,
        bias_factor_2d=bias_factor_2d,
        e_bin_edges=e_bin_edges,
        fallback_layer_mask=fallback,
        n_events_per_cell=n_events_per_cell,
        bias_factor_1d=bias_factor_1d,
        active_layer_mask=active_layer_mask,
        n_layers=nl,
        n_samples=5000,
        n_bins=nbins,
        bias_mode=bias_mode,
    )


class TestApplyBiasCorrection2D(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_layer_only_2d_matches_1d(self):
        """If bias_factor_2d is column-constant (same for all bins), output equals 1D."""
        nl, nbins = 11, 8
        rng = np.random.default_rng(0)
        bias_1d = rng.uniform(0.6, 1.4, size=nl)
        bias_2d = np.tile(bias_1d.reshape(-1, 1), (1, nbins))
        edges = np.linspace(0.0, 4.0, nbins + 1)
        npz = self.tmp_path / "bc.npz"
        _save_2d_npz(npz, bias_2d, edges, bias_factor_1d=bias_1d)

        n_events = 32
        gen = np.full((n_events, nl + 4), 100, dtype=np.int32)  # extra padding cols
        energies = rng.uniform(10**0.5, 10**3.5, size=n_events)
        out = apply_bias_correction_2d(gen, npz, energies)

        expected = np.round(100 * bias_1d).astype(np.int32)
        self.assertEqual(out.shape, gen.shape)
        np.testing.assert_array_equal(
            out[:, :nl], np.broadcast_to(expected, (n_events, nl))
        )
        np.testing.assert_array_equal(out[:, nl:], gen[:, nl:])  # padding untouched

    def test_bias_mode_assertion_rejects_1d_npz(self):
        """1D-style npz (no bias_mode key) must raise ValueError."""
        nl = 11
        npz = self.tmp_path / "bc1d.npz"
        np.savez(
            npz,
            bias_factor=np.ones(nl),
            n_layers=nl,
            active_layer_mask=np.ones(nl, dtype=bool),
        )
        gen = np.zeros((1, nl), dtype=np.int32)
        energies = np.array([100.0])
        with self.assertRaises(ValueError) as ctx:
            apply_bias_correction_2d(gen, npz, energies)
        self.assertIn("bias_mode", str(ctx.exception))

    def test_bias_mode_assertion_rejects_wrong_mode(self):
        """npz with bias_mode='1d' tag must also raise."""
        nl, nbins = 4, 2
        npz = self.tmp_path / "bc.npz"
        _save_2d_npz(
            npz,
            np.ones((nl, nbins)),
            np.linspace(0.0, 1.0, nbins + 1),
            bias_mode="1d",
        )
        gen = np.zeros((1, nl), dtype=np.int32)
        energies = np.array([10.0])
        with self.assertRaises(ValueError) as ctx:
            apply_bias_correction_2d(gen, npz, energies)
        self.assertIn("expected '2d'", str(ctx.exception))

    def test_fallback_layer_mask_factor_applied(self):
        """Cells flagged as fallback still apply whatever stored bias_factor_2d carries."""
        nl = 4
        bias_1d = np.array([1.5, 1.5, 1.5, 1.5])
        bias_2d = np.array(
            [
                [1.0, 1.5],
                [1.0, 1.5],
                [1.0, 1.5],
                [1.0, 1.5],
            ]
        )
        fallback = np.array(
            [
                [False, True],
                [False, True],
                [False, True],
                [False, True],
            ]
        )
        edges = np.array([0.0, 2.0, 4.0])
        npz = self.tmp_path / "bc.npz"
        _save_2d_npz(npz, bias_2d, edges, fallback=fallback, bias_factor_1d=bias_1d)

        gen = np.full((2, nl), 10, dtype=np.int32)
        # log10(10) = 1 → bin 0 (factor 1.0); log10(1000) = 3 → bin 1 (factor 1.5, fallback)
        energies = np.array([10.0, 1000.0])
        out = apply_bias_correction_2d(gen, npz, energies)
        self.assertEqual(int(out[0, 0]), 10)
        self.assertEqual(int(out[1, 0]), 15)

    def test_out_of_range_energy_clamps_and_warns(self):
        """Energies outside [edges[0], edges[-1]] clamp to first/last bin
        and emit a stderr warning."""
        nl, nbins = 4, 4
        bias_2d = np.tile(np.array([2.0, 1.0, 1.0, 0.5])[None, :], (nl, 1))
        edges = np.linspace(1.0, 5.0, nbins + 1)  # log10(E) in [1, 5]
        npz = self.tmp_path / "bc.npz"
        _save_2d_npz(npz, bias_2d, edges)

        gen = np.full((4, nl), 10, dtype=np.int32)
        # edges = [1,2,3,4,5]; np.digitize semantics: returns i s.t. edges[i-1] <= v < edges[i]
        # log10(0.1) = -1   → digitize=0  → -1 → clip → bin 0 (factor 2.0) → 20
        # log10(31.6228)≈1.5 → digitize=1 → 0  → bin 0 (factor 2.0) → 20
        # log10(31622.78)≈4.5 → digitize=4 → 3 → bin 3 (factor 0.5) → 5
        # log10(1e6) = 6     → digitize=5 → 4 → clip → bin 3 (factor 0.5) → 5
        energies = np.array([0.1, 31.6228, 31622.78, 1e6])
        buf = io.StringIO()
        with redirect_stderr(buf):
            out = apply_bias_correction_2d(gen, npz, energies)
        err_text = buf.getvalue()
        self.assertIn("WARNING", err_text)
        self.assertIn("1 events below", err_text)
        self.assertIn("1 above", err_text)
        self.assertEqual(int(out[0, 0]), 20)  # below clamped → bin 0
        self.assertEqual(int(out[1, 0]), 20)  # log10≈1.5 → bin 0
        self.assertEqual(int(out[2, 0]), 5)  # log10≈4.5 → bin 3
        self.assertEqual(int(out[3, 0]), 5)  # above clamped → bin 3


if __name__ == "__main__":
    unittest.main(verbosity=2)
