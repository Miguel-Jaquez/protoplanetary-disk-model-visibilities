MODEL_CONFIG = {
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

RUN_CONFIG = {
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
