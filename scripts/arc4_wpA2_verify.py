#!/usr/bin/env python3
"""Arc-4 WP-A2 verifier: corrective patch (P1-P5) + axis validity (A2-1..7). Supervisor-authored; DO NOT MODIFY.

Supersedes scripts/arc4_wpA_verify.py (changed by the PI, not by the executor:
test filenames aligned to what WP-A actually shipped; the floor/ceiling contract
for continuous axes was withdrawn as a PI design error; WP-A2 checks added).

Exit 0 == goal reached. Exit 1 == open checks. Exit 2 == integrity violation.
Flaggable checks pass only with an evidence-backed entry in results/arc4_wpA2/FLAGS.json.
Stdlib only. Run: python scripts/arc4_wpA2_verify.py
"""
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W1 = ROOT / "results" / "arc4_wpA"
W2 = ROOT / "results" / "arc4_wpA2"

FROZEN = {
    "experiment/preregistered/arc3_tierB_preregistration.md": "7678f89e650d7f3bc0634074abffe09b8c2f37dfc9ce1561e5f259f56039d5f9",
    "experiment/preregistered/cfg_sweep_predictions.md":      "3a883f153500e050952df13b69d9eead79cd9b271ab019ebfda0ece6fc1e518e",
    "experiment/preregistered/go_map_gate_language.md":       "872918c1d05385694e3639b7f9f245da653bbb75a8f5cdbd433835f13fa14fcf",
    "experiment/preregistered/stage_m_rerun_interpretations.md": "2a51aa9b9d036dc6e98869eb0846f09856c57cc98ba915d860e56ed21b08454b",
    "experiment/preregistered/f1_protocol_predictions.md":    "0ecb9435906e02df4442214a802840d000086ab096155916309c800bdfe07abc",
    "configs/thresholds.json":                                "66bbc9ca8714a46adabd353519a5ebdd7c71f4d1ad4bbc2840f935822af8bb70",
    "configs/axes.json":                                      "f8eb61c9e6c2fd01a9197952c3798263294da5b097223b015450fbf10e069a78",
    "configs/coarse_class_map.json":                          "55b5a1d4116caa4503a6b4b17192425da487a9c4385a287e343d850795be4fe7",
}

# Only these may be discharged by an evidence-backed FLAG (cached artifact may be absent).
FLAGGABLE = {"A2-4", "A2-6"}

results = []


def check(task, name, ok, detail=""):
    results.append((task, name, bool(ok), detail, task in FLAGGABLE))


def sha256(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def jload(p):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def txt(p):
    try:
        return p.read_text(errors="replace")
    except Exception:
        return ""


def run(cmd, timeout=2400):
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except Exception as e:  # noqa: BLE001
        return 99, str(e)


def cols(p):
    try:
        with p.open() as f:
            return set(next(csv.reader(f), []))
    except Exception:
        return set()


# ---------------------------------------------------------------- integrity
def c_integrity():
    bad = [rel for rel, want in FROZEN.items()
           if not (ROOT / rel).exists() or sha256(ROOT / rel) != want]
    check("C1", "frozen files unmodified (8 hashes)", not bad, "; ".join(bad))
    return not bad


def c_verifier_untouched():
    rc, out = run(["git", "log", "--oneline", "--", "scripts/arc4_wpA2_verify.py"])
    n = len([l for l in out.splitlines() if l.strip()])
    check("C7", "verifier committed once, never edited", n == 1, f"{n} commits")


# ------------------------------------------------------------ WP-A closeout
def c_portability():
    """The suite must be green on a checkout WITHOUT run artifacts, not only on the node.
    A fresh clone contains exactly the committed files, i.e. the public tree."""
    import shutil, tempfile
    d = tempfile.mkdtemp(prefix="pubcheck_")
    try:
        rc, out = run(["git", "clone", "--quiet", str(ROOT), d])
        if rc != 0:
            check("PORT", "fresh clone succeeds", False, out.strip()[:80]); return
        r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"],
                           cwd=d, capture_output=True, text=True, timeout=2400)
        o = (r.stdout or "") + (r.stderr or "")
        f_ = int(m.group(1)) if (m := re.search(r"(\d+) failed", o)) else 0
        e_ = int(m.group(1)) if (m := re.search(r"(\d+) error", o)) else 0
        p_ = int(m.group(1)) if (m := re.search(r"(\d+) passed", o)) else 0
        check("PORT", "suite green on an artifact-free checkout (no exact-float asserts)",
              f_ == 0 and e_ == 0, f"{p_} passed, {f_} failed, {e_} errors off-node")
    except Exception as ex:  # noqa: BLE001
        check("PORT", "portability run completed", False, str(ex)[:80])
    finally:
        shutil.rmtree(d, ignore_errors=True)


def c_closeout():
    # WP-A is merged to public main (d261b0d); no bundle needed. Only the durable
    # regression guards and the suite are checked here.
    for t in ("tests/test_phase3_labels.py", "tests/test_b4_bridge_stats.py",
              "tests/test_phase2_readout_aggregate.py"):
        check("CO", f"WP-A regression test still present: {t}", (ROOT / t).exists())
    rc, out = run([sys.executable, "-m", "pytest", "tests/", "-q"])
    p_ = int(m.group(1)) if (m := re.search(r"(\d+) passed", out)) else 0
    f_ = int(m.group(1)) if (m := re.search(r"(\d+) failed", out)) else 0
    e_ = int(m.group(1)) if (m := re.search(r"(\d+) error", out)) else 0
    check("CO", "pytest: zero failures, >= 1041 passed", f_ == 0 and e_ == 0 and p_ >= 1041,
          f"{p_} passed, {f_} failed, {e_} errors")

# ------------------------------------------- P: corrective patch (GPT WP-A.1)
def p1_docs():
    head = "\n".join(txt(ROOT / "results/PI_REPORT_arc3.md").splitlines()[:4]).upper()
    check("P1", "PI_REPORT_arc3.md carries a SUPERSEDED banner", "SUPERSEDED" in head)
    g = txt(ROOT / "PI_review_guidance.markdown")
    cs, a3 = g.find("CURRENT_STATUS.md"), g.find("PI_REPORT_arc3.md")
    check("P1", "guidance points at CURRENT_STATUS.md before PI_REPORT_arc3.md",
          cs != -1 and (a3 == -1 or cs < a3))
    check("P1", "guidance no longer presents DIAGNOSTIC-strong / F-1 refuted as current",
          "DIAGNOSTIC-strong" not in g and "**refuted**" not in g)


def p2_bridge():
    d = jload(W1 / "b4_bridge_corrected.json")
    if not d:
        check("P2", "b4_bridge_corrected.json readable", False)
        return
    jr = d.get("joint_recovery")
    check("P2", "joint_recovery is the float 0.0 (not None)",
          isinstance(jr, float) and abs(jr) < 1e-12, str(jr))
    check("P2", "citable_result == joint_recovery", d.get("citable_result") == "joint_recovery")
    means = d.get("mean_per_axis_recovery", {})
    check("P2", "no per-axis mean marked citable",
          all(v.get("citable") is False for v in means.values() if isinstance(v, dict)))
    check("P2", "joint bootstrap resamples the scalar floor", d.get("joint_floor_resampled") is True)
    check("P2", "joint result carries the simulated-flip qualifier",
          "simulated symmetric keep-flip" in txt(W1 / "b4_bridge_corrected.md"))
    check("P2", "regression test for the joint floor", (ROOT / "tests/test_b4_joint_floor.py").exists())


def p3_policy():
    check("P3", "phase4 header drops the false matched-scoring-call claim",
          "matched scoring-call" not in txt(ROOT / "scripts/phase4_policy.py"))
    check("P3", "AMD-22 records the ceiling-allocation accounting change",
          "AMD-22" in txt(ROOT / "experiment/preregistered/amendments_arc4.md"))
    check("P3", "'matched compute' renamed to rounded-up comparator",
          "rounded-up compute comparator" in txt(W1 / "policy_report_corrected.md"))


def p4_entropy():
    check("P4", "--legacy-arc3 flag exists", "--legacy-arc3" in txt(ROOT / "scripts/c_two_budgets.py"))
    d = jload(W2 / "entropy_lens_v3.json")
    check("P4", "abstain CIs are clip-bootstrapped, not label-level Wilson",
          bool(d) and d.get("ci_method") == "clip_bootstrap")
    check("P4", "default run does not regenerate the withdrawn narrative",
          bool(d) and "mode collapse" not in json.dumps(d).lower())


def p5_provenance():
    for f in ("tests_full.log", "tests_full.log.sha256"):
        check("P5", f"hashed full-suite log committed: {f}", (ROOT / f).exists())
    wf = ROOT / ".github" / "workflows"
    check("P5", "CI workflow for the artifact-independent subset",
          wf.exists() and bool(list(wf.glob("*.yml"))))
    check("P5", "CURRENT_STATUS cites canonical remote head d261b0d",
          "d261b0d" in txt(ROOT / "results/CURRENT_STATUS.md"))


def a2_2_class_proof():
    d = jload(W2 / "class_reconstruction.json")
    check("A2-2", "class_reconstruction.json exists", d is not None)
    if not d:
        return
    check("A2-2", "journal reconstruction reproduces the committed class CSV",
          d.get("reproduces_committed_csv") is True
          and isinstance(d.get("max_abs_delta"), (int, float)) and abs(d["max_abs_delta"]) <= 1e-9,
          f"delta={d.get('max_abs_delta')}")
    need = {"a_ind_confident", "a_ind_naive", "s_commit_confident", "s_commit_naive",
            "gap_confident", "gap_naive", "abstain_rate_by_s", "n_unscorable_cells"}
    check("A2-2", "naive vs confident published side by side",
          need <= set(d), str(sorted(need - set(d)))[:110])
    check("A2-2", "power check: naive and confident curves actually differ",
          d.get("naive_confident_differ") is True,
          "if identical, the cohort has no abstains and the test has no power")
    check("A2-2", "library path foley_cw/commitment.py uses confident_agreement",
          "confident_agreement" in txt(ROOT / "foley_cw/commitment.py"))
    check("A2-2", "regression test pins the library-path fix",
          (ROOT / "tests/test_commitment_abstain.py").exists())


# ------------------------------------------------------------------- WP-A2
AXES = {"presence", "timing", "class", "material"}


def a2_1_informativeness():
    d = jload(W2 / "axis_validity.json")
    check("A2-1", "axis_validity.json exists", d is not None)
    if not d:
        return
    per = d.get("per_axis", {})
    check("A2-1", "all four axes scored", AXES <= set(per), str(sorted(per)))
    need = {"majority_share", "k_eff", "a_between_video", "a_ind_mean", "verdict"}
    missing = {a: sorted(need - set(v)) for a, v in per.items() if not need <= set(v)}
    check("A2-1", "each axis reports majority/k_eff/between-video floor/A_ind/verdict",
          not missing, str(missing)[:120])
    ok = all(v.get("verdict") in ("INFORMATIVE", "DEGENERATE") for v in per.values())
    check("A2-1", "verdicts are INFORMATIVE|DEGENERATE", ok)
    check("A2-1", "thresholds declared before scoring", isinstance(d.get("thresholds"), dict))


def a2_3_partition():
    p = W2 / "determination_partition.csv"
    check("A2-3", "determination_partition.csv exists", p.exists())
    if p.exists():
        need = {"axis_id", "clip_id", "a_ind", "status", "s_commit"}
        c = cols(p)
        check("A2-3", "per-clip status columns present", need <= c, str(sorted(need - c)))
    d = jload(W2 / "window_partitioned.json")
    check("A2-3", "window_partitioned.json exists", d is not None)
    if not d:
        return
    per = d.get("per_axis", {})
    need = {"n_video_determined", "n_crossing", "n_censored", "km_median",
            "km_ci_lo", "km_ci_hi", "legacy_mean_crossers"}
    missing = {a: sorted(need - set(v)) for a, v in per.items() if not need <= set(v)}
    check("A2-3", "windows on crossing+censored only, with KM median + CI and legacy alongside",
          bool(per) and not missing, str(missing)[:120])
    bad = [a for a, v in per.items()
           if not isinstance(v.get("n_video_determined"), int)
           or v["n_video_determined"] + v.get("n_crossing", -1) + v.get("n_censored", -1) != 200]
    check("A2-3", "partition sums to 200 clips per axis", not bad, str(bad))


def a2_4_material():
    d = jload(W2 / "material_relative.json")
    check("A2-4", "material_relative.json exists", d is not None)
    if not d:
        return
    need = {"a_between_video_cosine", "relative_agreement", "chance_level",
            "s_commit_cosine", "s_commit_relative", "readout_cosine", "readout_relative"}
    check("A2-4", "nearest-reference rule reported vs legacy cosine",
          need <= set(d), str(sorted(need - set(d)))[:120])
    check("A2-4", "chance level is 0.5 (relative rule)", abs(float(d.get("chance_level", -1)) - 0.5) < 1e-9)


def a2_5_table():
    md = txt(W2 / "axis_diagnostics.md")
    check("A2-5", "axis_diagnostics.md exists", bool(md))
    if not md:
        return
    lo = md.lower()
    need = ["majority", "k_eff", "between", "abstain", "video_determined", "censored", "margin", "verdict"]
    miss = [k for k in need if k.replace("_", " ") not in lo and k not in lo]
    check("A2-5", "table carries all diagnostic columns", not miss, str(miss))
    check("A2-5", "all four axes present", all(a in lo for a in AXES))


def a2_6_swap():
    d = jload(W2 / "swap_final.json")
    check("A2-6", "swap_final.json exists", d is not None)
    if not d:
        return
    est = d.get("estimands", {})
    check("A2-6", "both estimands named with explicit denominators",
          {"unconditional", "donor_ne_source"} <= set(est), str(sorted(est)))
    dns = est.get("donor_ne_source", {})
    check("A2-6", "primary estimand has n + Clopper-Pearson CI",
          {"follow_only", "n", "ci_lo", "ci_hi"} <= set(dns), str(sorted(dns))[:100])
    check("A2-6", "material swap uses the relative rule",
          d.get("material_rule") == "nearest_reference")
    check("A2-6", "no floor/ceiling comparison across incompatible scales",
          "ceiling" not in json.dumps(d).lower() or d.get("scale_note"))


def a2_7_amd19():
    p = ROOT / "experiment/preregistered/amendments_arc4.md"
    if not p.exists():
        check("A2-7", "amendments_arc4.md exists (from WP-A)", False, "missing")
        return
    s = txt(p)
    check("A2-7", "AMD-19 recorded (swap estimands named; AMD-18 figure superseded)",
          "AMD-19" in s and "8/17" in s and "8/20" in s)
    sums = jload(ROOT / "experiment/preregistered/SHA256SUMS.json") or {}
    e = sums.get("_amendment_2026-07_arc4")
    got = e if isinstance(e, str) else (e or {}).get("sha256")
    check("A2-7", "amendment hash re-registered after AMD-19", got == sha256(p))


def a2_repro():
    r = jload(W2 / "REPRO.json")
    check("A2-R", "REPRO.json present", r is not None)
    if not r:
        return
    bad = []
    for rel, e in (r.get("regenerators") or {}).items():
        pth = ROOT / rel
        if not pth.exists():
            bad.append(f"{rel}: absent"); continue
        h = sha256(pth)
        if not (e.get("run1") == e.get("run2") == e.get("run3_hashseed_1") == h):
            bad.append(f"{rel}: run1/run2/run3(PYTHONHASHSEED=1)/on-disk differ")
    check("A2-R", "byte-identical across 3 runs incl. a different PYTHONHASHSEED", not bad,
          "; ".join(bad)[:120])
    idx = jload(W2 / "numbers_index.json")
    ok = isinstance(idx, list) and len(idx) >= 8 and all(
        {"claim", "value", "artifact", "key"} <= set(i) and (ROOT / i["artifact"]).exists()
        for i in idx)
    check("A2-R", "every headline number traced to artifact+key (>=8 entries)", ok,
          f"{len(idx) if isinstance(idx, list) else 0} entries")
    check("A2-R", "WPA2_REPORT.md written", (W2 / "WPA2_REPORT.md").exists())


def c_git():
    rc, out = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    check("C6", "on branch arc4-wpA2", out.strip() == "arc4-wpA2", out.strip())
    rc, out = run(["git", "status", "--porcelain"])
    check("C6", "working tree clean", out.strip() == "", out.strip()[:80])


def main():
    ok_int = c_integrity()
    c_verifier_untouched()
    c_closeout()
    c_portability()
    for fn in (p1_docs, p2_bridge, p3_policy, p4_entropy, p5_provenance,
               a2_1_informativeness, a2_2_class_proof, a2_3_partition, a2_4_material,
               a2_5_table, a2_6_swap, a2_7_amd19, a2_repro, c_git):
        fn()

    flags = {f["task"] for f in ((jload(W2 / "FLAGS.json") or {}).get("flags") or [])
             if all(str(f.get(k, "")).strip()
                    for k in ("task", "reason", "evidence_cmd", "evidence", "resolution"))}
    todo, flagged = [], []
    print(f"\n  {'TASK':7}{'CHECK':64}RESULT")
    print("-" * 96)
    for task, name, ok, detail, flaggable in results:
        if ok:
            st = "PASS"
        elif flaggable and task in flags:
            st = "FLAGGED"; flagged.append(task)
        else:
            st = "FAIL"; todo.append((task, name, detail))
        print(f"  {task:7}{name[:62]:64}{st}{('  — ' + detail) if detail and st != 'PASS' else ''}")
    print("-" * 96)
    if not ok_int:
        print("INTEGRITY VIOLATION: a frozen file changed. Restore it and stop.")
        return 2
    if todo:
        print(f"GOAL NOT REACHED — {len(todo)} open, {len(flagged)} flagged.")
        for t, n, d in todo:
            print(f"  TODO {t}: {n}{(' — ' + d) if d else ''}")
        print("\nFlaggable only: A2-4, A2-6 (missing cached artifacts, with evidence).")
        return 1
    print(f"GOAL REACHED — all checks pass ({len(flagged)} flagged with evidence).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
