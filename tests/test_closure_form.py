"""The closed forms behind `usd/_closure_lut.json`, verified against the shipped table itself.

These are NOT the path the robot runs. `ArmModel.passives` still interpolates the baked grid
bilinearly and must keep doing so: the grip/close/lift path was tuned against the interpolated
model, and switching it moves the hand by over a millimetre (pinned below). The closed forms are
here for the things a table cannot give -- an exact value between grid points and, above all, an
exact derivative.
    .venv/bin/python3 tests/test_closure_form.py"""
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from morph.arm.model import ArmModel  # noqa: E402

LUT_PATH = os.path.join(ROOT, "usd/_closure_lut.json")
M = ArmModel.load(os.path.join(ROOT, "usd/_arm_model.json"), LUT_PATH)
LUT = json.load(open(LUT_PATH))

# Reproduction of the baked table by the closed forms, measured over all 21x21 cells with the
# model-derived D2. Rotation in rad, slide in m.
CELL_TOL_RAD = 1e-5
CELL_TOL_M = 1e-6


def _domain(n=37, pad=0.0):
    """(dh, a1) sample pairs spanning the LUT domain, `pad` in from each edge."""
    dh = np.linspace(LUT["dh"][0] + pad, LUT["dh"][-1] - pad, n)
    a1 = np.linspace(LUT["a1"][0] + pad, LUT["a1"][-1] - pad, n)
    return [(float(x), float(y)) for x in dh for y in a1]


def test_D2_comes_from_the_model_and_agrees_with_the_baked_file():
    """The bearing separation is model geometry, not a literal: twice the x offset of
    `RotationLeftJoint_1`'s origin from the left column bearing. If a model edit ever moves that
    joint, this fails instead of silently re-scaling every closed form."""
    p0 = M.joints["RotationLeftJoint_1"]["localPos0"]
    assert M.D2 == 2 * abs(p0[0]), (M.D2, p0)
    assert abs(M.D2 - LUT["D2"]) < 5e-6, f"model D2 {M.D2!r} vs baked {LUT['D2']!r}"


def test_closed_form_reproduces_every_baked_cell():
    """The real verifier: the forms are compared against the SHIPPED artefact at all 441 cells,
    so they cannot pass by agreeing with themselves."""
    worst = {n: 0.0 for n in LUT["passives"]}
    for i, dh in enumerate(LUT["dh"]):
        for j, a1 in enumerate(LUT["a1"]):
            p = M.passives_closed_form(dh, a1)
            for k, n in enumerate(LUT["passives"]):
                worst[n] = max(worst[n], abs(p[n] - LUT["grid"][i][j][k]))
    assert worst["RotationLeftJoint_1"] < CELL_TOL_RAD, worst
    assert worst["ArmRightJoint_1"] < CELL_TOL_M, worst
    assert worst["ContactCylinderJoint_1_1"] == 0.0, worst
    assert worst["ContactCylinderJoint2_1"] == 0.0, worst


def test_both_contact_cylinders_are_zero_in_every_baked_cell():
    """Reproduced faithfully because the table says so. FLAGGED, NOT FIXED: independent evidence
    says the real robot holds both at 0.0445 rad (2.55 deg) throughout every baked trajectory.
    That is a separate discrepancy in the bake, and correcting it here would change the shipped
    kinematics -- which this work must not."""
    seen = {v for row in LUT["grid"] for cell in row for v in cell[2:]}
    assert seen == {0.0}, seen
    for dh, a1 in _domain(9):
        p = M.passives_closed_form(dh, a1)
        assert p["ContactCylinderJoint_1_1"] == 0.0 and p["ContactCylinderJoint2_1"] == 0.0


def test_the_analytic_derivative_matches_a_central_difference():
    h = 1e-6
    for dh, a1 in _domain(31, pad=1e-3):
        jac = M.passives_jacobian(dh, a1)
        up_dh = M.passives_closed_form(dh + h, a1)
        dn_dh = M.passives_closed_form(dh - h, a1)
        up_a1 = M.passives_closed_form(dh, a1 + h)
        dn_a1 = M.passives_closed_form(dh, a1 - h)
        for n in LUT["passives"]:
            fd = ((up_dh[n] - dn_dh[n]) / (2 * h), (up_a1[n] - dn_a1[n]) / (2 * h))
            assert abs(jac[n][0] - fd[0]) < 1e-5, (n, dh, a1, jac[n], fd)
            assert abs(jac[n][1] - fd[1]) < 1e-5, (n, dh, a1, jac[n], fd)


def test_the_rotation_gain_peaks_at_one_over_D2_at_dh_zero():
    """The prize. The repo's "measured peak 9.870 rad/m" is this formula sampled off-grid; the
    closed form gives the true supremum exactly, 1/D2 = 10.0 rad/m at dh = 0."""
    peak = M.passives_jacobian(0.0, 0.3)["RotationLeftJoint_1"][0]
    assert abs(peak - 1.0 / M.D2) < 1e-12, peak
    assert abs(peak - 10.0) < 1e-3, peak
    for dh, a1 in _domain(41):
        assert M.passives_jacobian(dh, a1)["RotationLeftJoint_1"][0] <= peak + 1e-12
    assert M.passives_jacobian(0.05, 0.3)["ArmRightJoint_1"][1] == -1.0


def test_the_offgrid_divergence_from_the_shipped_path_is_pinned():
    """Bilinear interpolation of `atan` is exact ON grid points and not between them. This is why
    `passives` was NOT switched to the closed form: the gap is real hand motion on a robot whose
    clearances were tuned against the interpolated model, and validating a swap needs the
    simulator. Pinned in BOTH directions -- the upper bound fails loudly if the table is ever
    re-baked, the lower bound fails if someone quietly points `passives` at the closed form."""
    rng = np.random.default_rng(0)
    worst = {n: 0.0 for n in LUT["passives"]}
    for dh, a1 in zip(rng.uniform(LUT["dh"][0], LUT["dh"][-1], 20000),
                      rng.uniform(LUT["a1"][0], LUT["a1"][-1], 20000)):
        interp = M.passives(float(dh), float(a1))
        exact = M.passives_closed_form(float(dh), float(a1))
        for n in LUT["passives"]:
            worst[n] = max(worst[n], abs(interp[n] - exact[n]))
    assert worst["RotationLeftJoint_1"] <= 3.3e-3, (
        f"interpolated-vs-closed-form rotation gap {worst['RotationLeftJoint_1']:.4e} rad exceeds "
        f"the measured 3.167e-03 rad (0.18 deg) -- the table changed; on the hand's 0.424 m lever "
        f"that measured gap is 1.34 mm of hand position, so re-measure before trusting either path")
    assert worst["ArmRightJoint_1"] <= 5.2e-4, worst
    assert worst["RotationLeftJoint_1"] > 1e-3, (
        f"only {worst['RotationLeftJoint_1']:.4e} rad apart -- `passives` looks like it now IS the "
        f"closed form; the shipped bilinear path must stay as baked until simulator validation")


def test_the_shipped_path_still_returns_the_baked_grid_exactly():
    """The whole point of this suite is that nothing about `passives` moved."""
    for i in (0, 7, 20):
        for j in (0, 13, 20):
            p = M.passives(LUT["dh"][i], LUT["a1"][j])
            for k, n in enumerate(LUT["passives"]):
                assert abs(p[n] - LUT["grid"][i][j][k]) < 1e-12, (i, j, n)


# no pytest in the venv? run direct.
if __name__ == "__main__":
    fails = 0
    for name in sorted(n for n in dir() if n.startswith("test_")):
        try:
            globals()[name]()
            print(f"PASS {name}")
        except Exception as e:
            fails += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{'all passed' if not fails else str(fails) + ' failed'}")
    sys.exit(1 if fails else 0)
