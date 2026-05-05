# LatentMAS-interp · analysis pipeline

Post-hoc analysis scripts for the activations produced by `final_run.py`.
Each script reads `<activations_dir>/<condition>/<task>/example_XXXX/*` and
writes results to `<activations_dir>/results/<exp_name>/`.

## Launch

```bash
# everything in dependency order (~10–30 min depending on N)
python -m analysis.run_all --activations_dir /workspace/activations

# a subset
python -m analysis.run_all --activations_dir /workspace/activations --only L,P,Q,report

# one experiment standalone
python -m analysis.exp_l_subspace --activations_dir /workspace/activations
```

## What each script does

| Script | ROADMAP | Reads | Writes |
|---|---|---|---|
| `report.py` | FATAL 3 headline check | metadata.json across all conditions | `results/report/{accuracy,compute,buckets}.csv`, `headline_check.json` |
| `exp_a_wa_mechanism.py` | A1–A5 | `wa_matrix.pt`, latent_thoughts | `exp_a/exp_a.json`, `wa_singular_values.npy` |
| `exp_b_patching.py` | B1–B3 | `activation_patching/*/patching.json` | `exp_b/exp_b.json`, recovery heatmaps |
| `exp_c_task_geometry.py` | C1–C6 | LMAS / single-agent / TextMAS post-W_a | `exp_c/exp_c.json`, `umap_latent_mas.png` |
| `exp_d_trajectory.py` | D1–D5 | LMAS post-W_a + correctness | `exp_d/exp_d.json` (per-(a,r) AUC heatmap, semantic dirs) |
| `exp_e_role.py` | E1–E5 | LMAS + TextMAS post-W_a | `exp_e/exp_e.json` |
| `exp_f_information.py` | F1–F4 | LMAS + TextMAS post-W_a | `exp_f/exp_f.json` (Kraskov MI, RD curve) |
| `exp_g_groupthink.py` | G1–G5 | LMAS + TextMAS post-W_a | `exp_g/exp_g.json` |
| `exp_h_faithfulness.py` | H2–H4 | LMAS `logitlens.json` | `exp_h/exp_h.json` |
| `exp_i_error_attribution.py` | I1–I4 | LMAS + TextMAS post-W_a | `exp_i/exp_i.json` |
| `exp_j_uncertainty.py` | J1–J4 | LMAS + single-agent post-W_a + text | `exp_j/exp_j.json` |
| `exp_k_redundancy.py` | K1–K4 | LMAS + TextMAS post-W_a | `exp_k/exp_k.json` |
| `exp_l_subspace.py` | L1, L4–L6 + L2 aggregation | LMAS post-W_a (discovery split) | **`exp_l/basis.pt`** + `exp_l.json` |
| `exp_m_wa_ablation.py` | M1–M5 | metadata across W_a-ablation conditions | `exp_m/exp_m.json` |
| `exp_n_sycophancy.py` | N1–N4 | LMAS + TextMAS post-W_a | `exp_n/exp_n.json` |
| `exp_o_layer_routing.py` | O1–O5 | `latent_per_layer.pt` (skipped if `--no_layer_hidden` was set) | `exp_o/exp_o.json` |
| `exp_p_probe.py` | P1–P5 | LMAS round-1 post-W_a + buckets + `exp_l/basis.pt` | **`exp_p/probe.joblib`**, **`exp_p/basis.pt`**, `exp_p.json` |
| `exp_q_gated.py` | Q1–Q3 + gating controls | `topk_gated`, `random_gated`, `confidence_gated` metadata | `exp_q/exp_q.json` |

## Two-pass workflow for the gated conditions

Exp P writes the probe artifacts that `final_run.py`'s `confidence_gated`
condition needs. The flow is:

```bash
# pass 1 — main collection (Stage 3 of run_runpod.sh; skip confidence_gated)
python final_run.py --output_dir /workspace/runs/full \
    --conditions all --tasks gsm8k arc_challenge mbppplus \
    --max_samples 500 --device cuda --latent_space_realign

# subspace + probe (offline, ~5 min)
python -m analysis.exp_l_subspace --activations_dir /workspace/runs/full
python -m analysis.exp_p_probe   --activations_dir /workspace/runs/full

# pass 2 — confidence_gated only (Q2)
# the run_condition() resolver now finds /workspace/runs/full/exp_p/probe.joblib
python final_run.py --output_dir /workspace/runs/full \
    --conditions confidence_gated --tasks gsm8k arc_challenge mbppplus \
    --max_samples 200 --device cuda --latent_space_realign

# rerun analysis (now includes Q2 paired comparisons)
python -m analysis.run_all --activations_dir /workspace/runs/full --only Q,report
```

## Notes / known limitations

- **Bucketing prerequisite.** Buckets are written by `final_run.py` to
  `<root>/buckets/<task>.json` once both `latent_mas` and
  `single_agent_latent_greedy` have run. Several experiments (L Method B, P,
  Q per-bucket) skip silently if buckets are missing.
- **Splits.** `data/splits/{task}_{discovery,validation,test}.json` is
  written on the first run that calls `lock_splits()` in `final_run.py`.
  If you run a single condition before splits are locked (e.g., a smoke
  test), the splits may not match the full-collection ordering — re-lock by
  deleting `data/splits/` and rerunning the main collection.
- **`exp_o` requires `latent_per_layer.pt`.** The `--no_layer_hidden` flag
  in `run_runpod.sh` disables this for the full collection. Drop the flag
  if you want Exp O on the headline data.
- **Statistical conventions** (ROADMAP Part 3): every reported number has a
  95% CI (Wilson for proportions, bootstrap or DeLong elsewhere); paired
  binary comparisons use McNemar; AUC comparisons use DeLong; FDR is
  Benjamini–Hochberg where >5 hypotheses are tested simultaneously.
