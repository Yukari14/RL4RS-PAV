"""Shared CLI helpers for online PAV experiments."""
import argparse

from rl4rs.pav.prover import format_prover_banner


def add_pav_cli_args(parser):
    parser.add_argument("--pav-trial-name", default="a_50k_logged")
    parser.add_argument("--pav-suffix", default="pav_v3")
    parser.add_argument("--pav-alpha", type=float, default=None, help="Override PAV alpha")
    parser.add_argument("--pav-variant", default=None,
                        help="Tag for output filenames, e.g. noverifier / lowalpha")
    parser.add_argument("--no-verifier", action="store_true",
                        help="Disable verifier gate (use progress only, ablation)")
    parser.add_argument("--prover-kind", default=None,
                        help="Override prover_kind recorded in stats (logging, random, uniform, ...)")
    parser.add_argument("--verifier-output-mode", default=None,
                        choices=["binary", "q_regression", "dual"],
                        help="Verifier head mode for online artifact loading")


def apply_pav_cli_overrides(pav_config, args):
    """Apply optional CLI overrides onto a PAVConfig instance."""
    if getattr(args, "pav_alpha", None) is not None:
        pav_config.alpha = float(args.pav_alpha)
    if getattr(args, "no_verifier", False):
        pav_config.use_verifier = False
        pav_config.use_raw_progress = True
    else:
        pav_config.use_verifier = True
        pav_config.use_raw_progress = False
    if getattr(args, "prover_kind", None):
        pav_config.prover_kind = str(args.prover_kind)
    if getattr(args, "verifier_output_mode", None):
        pav_config.verifier_output_mode = str(args.verifier_output_mode)
    return pav_config


def print_pav_config_banner(pav_config):
    for line in format_prover_banner(pav_config):
        print(line, flush=True)


def pav_condition_tag(args, use_pav=True):
    if not use_pav:
        return "raw"
    if getattr(args, "pav_variant", None):
        return "pav_{}".format(args.pav_variant)
    suffix = getattr(args, "pav_suffix", None) or "pav_v3"
    parts = [suffix]
    if getattr(args, "no_verifier", False):
        parts.append("noverifier")
    if getattr(args, "pav_alpha", None) is not None:
        parts.append("a{}".format(args.pav_alpha))
    return "_".join(parts)
