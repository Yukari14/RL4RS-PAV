# PAV Discussion Notes (limitations & assumptions)

For paper Discussion section. Eval metrics must always use **raw simulator reward**, not shaped reward.

## Assumptions

1. **Frozen reward model**: $R_\phi$ is fit offline on logged data; online rollouts do not refit unless explicitly enabled.
2. **Non–potential-based shaping**: $r'_t$ depends on action via Verifier; optimal policy invariance (Ng et al.) does **not** hold.
3. **Streaming progress**: online step-wise shaping uses horizon=1 progress unless episode-end relabeling is added.

## Known limitations

- Error propagation from $R_\phi$ with no uncertainty quantification.
- Verifier sign labels are heuristic; high-variance returns add label noise (mitigation: `verifier_label_mode=magnitude` at refit time).
- Hyperparameters $\alpha$, $k$, clip are empirical.

## Recommended ablations

- raw RL vs PAV vs `--no-verifier` vs `--pav-alpha` sweep.
