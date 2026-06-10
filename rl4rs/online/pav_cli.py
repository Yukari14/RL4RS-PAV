"""Shared CLI helpers for online PAV experiments."""
import argparse


def add_pav_cli_args(parser):
    parser.add_argument("--pav-trial-name", default="a_50k_logged")
    parser.add_argument("--pav-suffix", default="pav")
    parser.add_argument("--pav-alpha", type=float, default=None, help="Override PAV alpha")
    parser.add_argument("--pav-variant", default=None,
                        help="Tag for output filenames, e.g. gated / noverifier")
    parser.add_argument("--no-verifier", action="store_true",
                        help="Disable verifier (use raw progress only)")
    parser.add_argument("--confidence-gating", action="store_true",
                        help="Scale alpha by verifier confidence (default in config)")
    parser.add_argument("--no-confidence-gating", action="store_true",
                        help="Disable confidence gating (legacy behavior)")
    parser.add_argument("--max-shaping-ratio", type=float, default=None,
                        help="Cap |shaped-raw| vs max(ratio*|raw|, floor)")


def apply_pav_cli_overrides(pav_config, args):
    """Apply optional CLI overrides onto a PAVConfig instance."""
    if getattr(args, "pav_alpha", None) is not None:
        pav_config.alpha = float(args.pav_alpha)
    if getattr(args, "no_verifier", False):
        pav_config.use_verifier = False
        pav_config.use_raw_progress = True
    elif getattr(args, "raw_progress", False):
        pav_config.use_raw_progress = True
    if getattr(args, "no_confidence_gating", False):
        pav_config.confidence_gating = False
    elif getattr(args, "confidence_gating", False):
        pav_config.confidence_gating = True
    if getattr(args, "max_shaping_ratio", None) is not None:
        pav_config.max_shaping_ratio = float(args.max_shaping_ratio)
    return pav_config


def pav_condition_tag(args, use_pav=True):
    if not use_pav:
        return "raw"
    if getattr(args, "pav_variant", None):
        return "pav_{}".format(args.pav_variant)
    parts = ["pav"]
    if getattr(args, "no_verifier", False):
        parts.append("noverifier")
    if getattr(args, "no_confidence_gating", False):
        parts.append("legacy")
    if getattr(args, "pav_alpha", None) is not None:
        parts.append("a{}".format(args.pav_alpha))
    return "_".join(parts) if len(parts) > 1 else "pav"
