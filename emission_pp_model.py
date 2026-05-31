# -*- coding: utf-8 -*-
"""Command line runner for protoplanetary disk emission models.

The model configuration is read from a directory, so this script can be
launched from any current working directory:

    python /path/to/emission_pp_model.py --config-dir /path/to/config_dir
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np

from ppdisk_components import (
    apply_normalization_to_model_components,
    build_intrinsic_emission_model,
    convolve_convert_and_add_noise,
    normalize_model_image,
    plot_image,
)


CONFIG_FILENAMES = (
    "model_config.py",
    "config.py",
    "model_config.json",
    "config.json",
)


DEFAULT_RUN_CONFIG = {
    "normalization": {
        "method": "total_flux",
        "value": 1.0,
    },
    "observation": {
        "enabled": True,
        "distance_pc": 140.0,
        "bmaj_arcsec": 0.0036,
        "bmin_arcsec": 0.0036,
        "bpa_deg": 0.0,
        "noise_std_jybeam": 1.85e-6,
        "random_seed": 12345,
    },
    "output": {
        "directory": "outputs",
        "write_fits": True,
        "make_plots": False,
    },
}


DEFAULT_MODEL_CONFIG = {
    "geometry": {
        "nx": 500,
        "ny": 500,
        "pixscale_au": 0.1,
        "inc_deg": 0.0,
        "pa_deg": 0.0,
        "x0_au": 0.0,
        "y0_au": 0.0,
    },
    "axisymmetric_disk": {
        "constant_disks": [
            {
                "enabled": True,
                "name": "uniform_disk_r15au",
                "amp": 1.0,
                "r_in_au": 0.0,
                "r_out_au": 5.0,
                "edge_smoothing_au": 1.0,
            }
        ],
        "rings": [
            {
                "enabled": True,
                "name": "bright_inner_ring",
                "amp": 1.0,
                "r0_au": 2.0,
                "sigma_au": 0.5,
            },
            {
                "enabled": True,
                "name": "ring1",
                "amp": 1.0,
                "r0_au": 7.0,
                "sigma_au": 0.5,
            },
            {
                "enabled": True,
                "name": "ring2",
                "amp": 1.0,
                "r0_au": 8.5,
                "sigma_au": 0.5,
            },
        ],
    },
    "planets": [
        {
            "enabled": True,
            "name": "planet_1_inner",
            "amp_rel": 0.6,
            "amp_rel_to": "local_disk",
            "r_au": 3.5,
            "phi_deg": 45.0,
            "sigma_r_au": 0.15,
            "sigma_phi_au": 0.15,
            "theta_deg": 0.0,
        },
        {
            "enabled": True,
            "name": "planet_2_inner_rings",
            "amp_rel": 0.8,
            "amp_rel_to": "local_disk",
            "r_au": 7.8,
            "phi_deg": 210.0,
            "sigma_r_au": 0.15,
            "sigma_phi_au": 0.15,
            "theta_deg": 0.0,
        },
    ],
    "ring_asymmetries": [],
}


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_path(path: str | Path, base_dir: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


def _find_config_file(config_dir: Path, config_file: str | None) -> Path:
    if config_file:
        path = _resolve_path(config_file, config_dir)
        if not path.exists():
            raise FileNotFoundError(f"No existe el archivo de config: {path}")
        return path

    for filename in CONFIG_FILENAMES:
        path = config_dir / filename
        if path.exists():
            return path

    expected = ", ".join(CONFIG_FILENAMES)
    raise FileNotFoundError(
        f"No encontre un config en {config_dir}. Nombres esperados: {expected}."
    )


def _load_python_config(path: Path) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("emission_model_user_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No pude importar el config Python: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model_config = getattr(module, "MODEL_CONFIG", None)
    if model_config is None:
        model_config = getattr(module, "model_config", None)
    if model_config is None:
        raise ValueError(
            f"{path} debe definir una variable MODEL_CONFIG o model_config."
        )

    run_config = getattr(module, "RUN_CONFIG", None)
    if run_config is None:
        run_config = getattr(module, "run_config", {})

    return {
        "model_config": model_config,
        "run_config": run_config,
    }


def _load_json_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if "model_config" in data:
        model_config = data["model_config"]
        run_config = {
            key: value
            for key, value in data.items()
            if key not in {"model_config", "MODEL_CONFIG"}
        }
    else:
        model_config = data
        run_config = {}

    return {
        "model_config": model_config,
        "run_config": run_config,
    }


def load_config(config_dir: str | Path, config_file: str | None = None) -> dict[str, Any]:
    config_dir = Path(config_dir).expanduser().resolve()
    if not config_dir.is_dir():
        raise NotADirectoryError(f"No es un directorio de config: {config_dir}")

    path = _find_config_file(config_dir, config_file)
    if path.suffix == ".py":
        loaded = _load_python_config(path)
    elif path.suffix == ".json":
        loaded = _load_json_config(path)
    else:
        raise ValueError("El config debe ser .py o .json.")

    run_config = _deep_update(DEFAULT_RUN_CONFIG, loaded.get("run_config", {}))
    return {
        "config_dir": config_dir,
        "config_file": path,
        "model_config": loaded["model_config"],
        "run_config": run_config,
    }


def write_fits_image(path: Path, image: np.ndarray) -> None:
    from astropy.io import fits

    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(image).writeto(path, overwrite=True)


def run_from_config(config: dict[str, Any]) -> dict[str, Any]:
    model_config = config["model_config"]
    run_config = config["run_config"]
    config_dir = config["config_dir"]

    output_dir = _resolve_path(run_config["output"]["directory"], config_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_intrinsic_emission_model(model_config)
    coords = model["coords"]

    norm_cfg = run_config.get("normalization")
    if norm_cfg:
        norm_result = normalize_model_image(
            img=model["img_total"],
            method=norm_cfg["method"],
            value=norm_cfg["value"],
        )
        model = apply_normalization_to_model_components(model, norm_result)
        img_jypix = model["img_total_normalized"]
    else:
        img_jypix = model["img_total"]

    products = {
        "model": model,
        "img_jypix": img_jypix,
    }

    if run_config["output"].get("make_plots", False):
        plot_image(
            img_jypix,
            coords,
            title=str(output_dir / "img_intrinsic"),
            unit_label="Jy/pixel",
            plane="sky",
        )

    if run_config["output"].get("write_fits", True):
        write_fits_image(output_dir / "img_jypix.fits", img_jypix)

    obs_cfg = run_config.get("observation", {})
    if obs_cfg.get("enabled", False):
        obs_result = convolve_convert_and_add_noise(
            img_jypix=img_jypix,
            pixscale_au=coords["pixscale_au"],
            distance_pc=obs_cfg["distance_pc"],
            bmaj_arcsec=obs_cfg["bmaj_arcsec"],
            bmin_arcsec=obs_cfg["bmin_arcsec"],
            bpa_deg=obs_cfg.get("bpa_deg", 0.0),
            noise_std_jybeam=obs_cfg["noise_std_jybeam"],
            random_seed=obs_cfg.get("random_seed"),
            return_kernel=True,
        )
        products["observation"] = obs_result

        image_jyarcsec2 = img_jypix / (
            (coords["pixscale_au"] / obs_cfg["distance_pc"]) ** 2
        )
        products["img_jyarcsec2"] = image_jyarcsec2

        if run_config["output"].get("make_plots", False):
            plot_image(
                image_jyarcsec2,
                coords,
                title=str(output_dir / "img_jyarcsec"),
                unit_label="Jy/arcsec^2",
                plane="sky",
            )
            plot_image(
                obs_result["img_convolved_jypix"],
                coords,
                title=str(output_dir / "img_convolved_jypix"),
                unit_label="Jy/pixel",
                plane="sky",
            )
            plot_image(
                obs_result["img_convolved_jybeam"],
                coords,
                title=str(output_dir / "img_convolved_jybeam"),
                unit_label="Jy/beam",
                plane="sky",
            )
            plot_image(
                obs_result["img_noisy_jybeam"],
                coords,
                title=str(output_dir / "img_convolved_jybeam_noisy"),
                unit_label="Jy/beam",
                plane="sky",
            )

        if run_config["output"].get("write_fits", True):
            write_fits_image(output_dir / "img_jyarcsec.fits", image_jyarcsec2)
            write_fits_image(
                output_dir / "img_convolved_jypix.fits",
                obs_result["img_convolved_jypix"],
            )
            write_fits_image(
                output_dir / "img_convolved_jybeam.fits",
                obs_result["img_convolved_jybeam"],
            )
            write_fits_image(
                output_dir / "img_convolved_jybeam_noisy.fits",
                obs_result["img_noisy_jybeam"],
            )

        print("Factor Jy/pixel -> Jy/beam:", obs_result["conversion_factor"])
        print("Omega_pix [arcsec^2]:", obs_result["omega_pix_arcsec2"])
        print("Omega_beam [arcsec^2]:", obs_result["omega_beam_arcsec2"])

    print(f"Config: {config['config_file']}")
    print(f"Outputs: {output_dir}")
    return products


def write_default_config(config_dir: str | Path) -> Path:
    config_dir = Path(config_dir).expanduser().resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "model_config.py"

    content = (
        "MODEL_CONFIG = "
        + repr(DEFAULT_MODEL_CONFIG)
        + "\n\nRUN_CONFIG = "
        + repr(DEFAULT_RUN_CONFIG)
        + "\n"
    )
    path.write_text(content, encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an emission model from a config directory."
    )
    parser.add_argument(
        "--config-dir",
        default=".",
        help=(
            "Directory containing model_config.py, config.py, "
            "model_config.json or config.json."
        ),
    )
    parser.add_argument(
        "--config-file",
        default=None,
        help="Optional config filename/path. Relative paths are resolved from --config-dir.",
    )
    parser.add_argument(
        "--write-default-config",
        action="store_true",
        help="Write an example model_config.py into --config-dir and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.write_default_config:
        path = write_default_config(args.config_dir)
        print(f"Config de ejemplo escrito en: {path}")
        return

    config = load_config(args.config_dir, args.config_file)
    run_from_config(config)


if __name__ == "__main__":
    main()
