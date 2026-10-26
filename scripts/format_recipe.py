#!/usr/bin/env python3
"""
Regroups the top-level keys of recipe .yaml files into sections (material and data generation, load steps and seeds,
extraction, distillation, FEM validation, other), with a heading comment per section. Works on the text: inline
comments, nested blocks and the header comment block are kept; old section-heading comment lines between keys are
replaced by the new headings; commented-out options ("# key: value") are kept and sorted like keys. The parsed
content is checked to be identical before writing.

Usage: python3 scripts/format_recipe.py configs/recipes_v2/*.yaml [--check]   (--check: only report files that change)
"""
import re
import sys

import yaml

RULE = "# " + "=" * 78
GROUPS = [
    ("Material and data generation", [
        "material_model_name", "material_params", "has_ground_truth", "dataset_path", "geometry_train",
        "geometry_val", "mesh_size", "control_mode", "stress_mode", "plane_stress_mode", "n_loadsteps",
        "target_load_true_top", "target_load_holes", "asym_factor", "disp_noise", "load_noise"]),
    ("Load steps and seeds", [
        "train_load_steps_indices", "val_load_steps_indices", "test_load_steps_indices", "seeds"]),
    ("Extraction (HSGP)", [
        # likelihood and free-residual noise
        "hsgp_likelihood", "hsgp_force_noise_rel", "hsgp_cv_folds", "hsgp_cv_tol",
        "reaction_components", "reaction_noise_rel", "reaction_noise_floor", "reaction_zero_components",
        # basis and hyperparameters
        "hsgp_num_basis_dev", "hsgp_num_basis_vol", "hsgp_box_factor",
        "hsgp_amplitude_prior_scale", "hsgp_lengthscale_prior_scale", "hsgp_sigma_cap", "hsgp_hyper_restarts"]),
    ("Distillation", [
        "distilled_material_model", "distill_target", "distillation_n_iterations",
        "dev_vol_distillation_n_iterations", "aniso_distillation_n_iterations", "sample_mode",
        "distillation_sampling_mode", "num_points", "num_func_samples", "max_gamma", "do_sensitivity",
        "sobol_threshold", "sobol_samples_factor", "dev_params_disabled", "vol_params_disabled",
        "aniso_params_disabled"]),
    ("FEM forward validation", ["run_validation", "val_number_samples", "val_workers"]),
]
KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:")
HEADING_RE = re.compile(r"^# --- .* -+$")
COMMENTED_KEY_RE = re.compile(r"^#\s*([A-Za-z_][A-Za-z0-9_]*)\s*:")      # a commented-out option, kept


def group_of(key):
    key = key.lstrip("#")
    for gi, (_, keys) in enumerate(GROUPS):
        if key in keys:
            return gi, keys.index(key)
    if key.startswith("hsgp_"):                      # unknown extraction options: end of the extraction group
        return 2, 10_000
    if "distill" in key or key.startswith("sobol"):
        return 3, 10_000
    if key.startswith("val_") or key.startswith("fem_"):
        return 4, 10_000
    return len(GROUPS), 0                            # other


def split(text):
    """(header lines, {key: block lines}) ; a block is the key line plus its continuation lines."""
    lines = text.rstrip("\n").split("\n")
    i, header = 0, []
    while i < len(lines) and not KEY_RE.match(lines[i]):
        header.append(lines[i])
        i += 1
    blocks, order, cur = {}, [], None
    for line in lines[i:]:
        m = KEY_RE.match(line)
        if m:
            cur = m.group(1)
            blocks[cur] = [line]
            order.append(cur)
        elif line.startswith((" ", "\t", "- ")) or line == "-":
            blocks[cur].append(line)                  # nested mapping / list items of the current key
        elif COMMENTED_KEY_RE.match(line):
            cur = "#" + COMMENTED_KEY_RE.match(line).group(1)
            while cur in blocks:
                cur += "#"
            blocks[cur] = [line]
            order.append(cur)
        elif line.strip() == "" or line.lstrip().startswith("#"):
            continue                                  # blank lines and old section headings are rebuilt
        else:
            blocks[cur].append(line)
    # drop this script's own section headings from the header (re-runs), then trailing blank lines
    header = [h for h in header if not HEADING_RE.match(h)]
    while header and header[-1].strip() == "":
        header.pop()
    return header, blocks, order


def format_text(text):
    header, blocks, order = split(text)
    ranked = sorted(order, key=lambda k: (*group_of(k), order.index(k)))
    out = list(header)
    current = None
    names = [g[0] for g in GROUPS] + ["Other"]
    for k in ranked:
        g = group_of(k)[0]
        if g != current:
            out += ["", f"# --- {names[g]} " + "-" * max(3, 72 - len(names[g]))]
            current = g
        out += blocks[k]
    return "\n".join(out).lstrip("\n") + "\n"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check = "--check" in sys.argv
    for path in args:
        text = open(path).read()
        new = format_text(text)
        if yaml.safe_load(new) != yaml.safe_load(text):
            print(f"SKIPPED (content would change): {path}")
            continue
        if new == text:
            continue
        if check:
            print(f"would reformat: {path}")
        else:
            open(path, "w").write(new)
            print(f"reformatted: {path}")


if __name__ == "__main__":
    main()
