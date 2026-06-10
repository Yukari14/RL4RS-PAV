"""Shared CLI helpers for online PAV experiments."""
import argparse


def add_pav_cli_args(parser):
    parser.add_argument("--pav-trial-name", default="a_50k_logged")
    parser.add_argument("--pav-suffix", default="pav_v2")
    parser.add_argument("--pav-alpha", type=float, default=None, help="Override PAV alpha")
    parser.add_argument("--pav-variant", default=None,
                        help="Tag for output filenames, e.g. noverifier / lowalpha")
    parser.add_argument("--no-verifier", action="store_true",
                        help="Disable verifier gate (use progress only, ablation)")


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
    return pav_config


def pav_condition_tag(args, use_pav=True):
    if not use_pav:
        return "raw"
    if getattr(args, "pav_variant", None):
        return "pav_{}".format(args.pav_variant)
    parts = ["pav_v2"]
    if getattr(args, "no_verifier", False):
        parts.append("noverifier")
    if getattr(args, "pav_alpha", None) is not None:
        parts.append("a{}".format(args.pav_alpha))
    return "_".join(parts)
