# AdaptiveGuard

**Razorpay Buildathon — Track 2 (AI Risk Manager). Loss class: serial + coordinated return abuse.**

> Given the SAME limited review/labeling budget, does an additional hidden
> relational view actually outperform simply retraining the exposed
> behavioral model under adaptive return-abuse behavior?

This is a defense-only evaluation harness, not a claim of a new algorithm.
The exposed/hidden disagreement mechanism is adapted from Sethi & Kantardzic
(2018, Predict-Detect); the adaptation framing draws on performative
prediction (Perdomo et al. 2020) and strategic classification (Levanon &
Rosenfeld 2021; Horowitz & Rosenfeld 2023). The contribution here is the
fair-budget comparison itself, run on a return-abuse-specific synthetic
environment with hard negatives and imperfect/delayed labels.

## Status

- [x] Phase 0 — synthetic environment (`data_gen/`) + entity-aware/temporal split
- [ ] Phase 1 — Model A + held-out metrics + cost
- [ ] Phase 2 — adaptive rounds, demonstrate Model A degradation
- [ ] Phase 3 — Model B + disagreement
- [ ] Phase 4 — retraining loop + fair-budget ablation (**hero experiment**)
- [ ] Phase 5 — noisy/delayed observed labels + review router
- [ ] Phase 6 — LLM investigator + API (P1)
- [ ] Phase 7 — architecture diagram, pitch script, final writeup

## Defense-only statement

The adaptive-population component (`data_gen/generate.py`) is a closed,
synthetic, offline evaluation harness. It never connects to production
systems, never probes real fraud controls, and generates no actionable
real-world evasion instructions. It exists solely to measure whether a
detector's performance degrades when the scored population reacts to it.

## Reproduce Phase 0

```bash
pip install -r requirements.txt
python data_gen/generate.py --config configs/config.yaml --out data/
python data_gen/split.py --data data/ --config configs/config.yaml
```
