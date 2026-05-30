# MODELO
# DISCO CENTRAL + ANILLOS + PLANETAS + ASIMETRÍAS AZIMUTALES

import numpy as np
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive/')

import os
# GEOMETRÍA DEL MODELO
def build_coordinate_grids(geometry):
    """
    Construye las coordenadas del plano del cielo y del plano
    intrínseco del disco.
    """

    nx = geometry["nx"]
    ny = geometry["ny"]

    pixscale_au = geometry["pixscale_au"]

    inc_deg = geometry["inc_deg"]
    pa_deg = geometry["pa_deg"]

    x0_au = geometry.get("x0_au", 0.0)
    y0_au = geometry.get("y0_au", 0.0)

    inc_rad = np.deg2rad(inc_deg)
    pa_rad = np.deg2rad(pa_deg)

    cosi = np.cos(inc_rad)

    if np.isclose(cosi, 0.0):
        raise ValueError("inc_deg demasiado cercano a 90 deg; cos(i) ~ 0.")

    x_1d = (np.arange(nx) - (nx - 1) / 2.0) * pixscale_au
    y_1d = (np.arange(ny) - (ny - 1) / 2.0) * pixscale_au

    x_sky, y_sky = np.meshgrid(x_1d, y_1d)

    x_sky_shifted = x_sky - x0_au
    y_sky_shifted = y_sky - y0_au

    x_rot =  x_sky_shifted * np.cos(pa_rad) + y_sky_shifted * np.sin(pa_rad)
    y_rot = -x_sky_shifted * np.sin(pa_rad) + y_sky_shifted * np.cos(pa_rad)

    x_disk = x_rot
    y_disk = y_rot / cosi

    r_disk = np.sqrt(x_disk**2 + y_disk**2)
    phi_disk = np.arctan2(y_disk, x_disk)

    return {
        "nx": nx,
        "ny": ny,
        "pixscale_au": pixscale_au,
        "inc_deg": inc_deg,
        "pa_deg": pa_deg,
        "inc_rad": inc_rad,
        "pa_rad": pa_rad,
        "cosi": cosi,
        "x0_au": x0_au,
        "y0_au": y0_au,
        "x_1d": x_1d,
        "y_1d": y_1d,
        "x_sky": x_sky,
        "y_sky": y_sky,
        "x_sky_shifted": x_sky_shifted,
        "y_sky_shifted": y_sky_shifted,
        "x_rot": x_rot,
        "y_rot": y_rot,
        "x_disk": x_disk,
        "y_disk": y_disk,
        "r_disk": r_disk,
        "phi_disk": phi_disk,
    }


# COMPONENTES AXISIMÉTRICAS
# ============================================================
# DISCO DE INTENSIDAD CONSTANTE
# ============================================================
def constant_intensity_disk_component(r_disk,
                                      amp=1.0,
                                      r_in_au=0.0,
                                      r_out_au=100.0,
                                      edge_smoothing_au=0.0):
    """
    Construye un disco con intensidad constante entre r_in_au y r_out_au.

    I(r) = amp,  si r_in <= r <= r_out
    I(r) = 0,    fuera de ese intervalo

    Si edge_smoothing_au > 0, los bordes se suavizan con funciones tanh.
    """

    if r_in_au < 0:
        raise ValueError("r_in_au debe ser >= 0.")

    if r_out_au <= r_in_au:
        raise ValueError("r_out_au debe ser mayor que r_in_au.")

    if edge_smoothing_au < 0:
        raise ValueError("edge_smoothing_au debe ser >= 0.")

    if edge_smoothing_au == 0.0:
        comp = np.zeros_like(r_disk)
        mask = (r_disk >= r_in_au) & (r_disk <= r_out_au)
        comp[mask] = amp
        return comp

    # Borde interno
    # Si r_in_au = 0, no debe existir depresión central.
    if np.isclose(r_in_au, 0.0):
        inner_edge = np.ones_like(r_disk)
    else:
        inner_edge = 0.5 * (
            1.0 + np.tanh((r_disk - r_in_au) / edge_smoothing_au)
        )

    # Borde externo
    outer_edge = 0.5 * (
        1.0 - np.tanh((r_disk - r_out_au) / edge_smoothing_au)
    )

    comp = amp * inner_edge * outer_edge

    return comp

def central_gaussian_component(r_disk, amp, sigma_au):
    """
    Componente central gaussiana:
        I(r) = amp * exp[-0.5 * (r / sigma)^2]
    """
    if sigma_au <= 0:
        raise ValueError("sigma_au de la componente central debe ser > 0.")

    return amp * np.exp(-0.5 * (r_disk / sigma_au)**2)


def gaussian_ring_component(r_disk, amp, r0_au, sigma_au):
    """
    Anillo gaussiano:
        I(r) = amp * exp[-0.5 * ((r-r0)/sigma)^2]
    """
    if sigma_au <= 0:
        raise ValueError("sigma_au del anillo debe ser > 0.")

    return amp * np.exp(-0.5 * ((r_disk - r0_au) / sigma_au)**2)


def make_axisymmetric_disk(axisymmetric_disk, coords):
    """
    Construye el disco axisimétrico completo:
    - discos de intensidad constante
    - componente central gaussiana
    - suma de anillos gaussianos
    """

    r_disk = coords["r_disk"]

    img_total = np.zeros_like(r_disk)

    img_constant_disks_total = np.zeros_like(r_disk)
    img_central = np.zeros_like(r_disk)
    img_rings_total = np.zeros_like(r_disk)

    constant_disk_components = {}
    ring_components = {}

    # ============================================================
    # Discos de intensidad constante
    # ============================================================
    constant_disks = axisymmetric_disk.get("constant_disks", [])

    for i, disk in enumerate(constant_disks):
        if not disk.get("enabled", True):
            continue

        disk_name = disk.get("name", f"constant_disk_{i+1}")

        disk_map = constant_intensity_disk_component(
            r_disk=r_disk,
            amp=disk.get("amp", 1.0),
            r_in_au=disk.get("r_in_au", 0.0),
            r_out_au=disk["r_out_au"],
            edge_smoothing_au=disk.get("edge_smoothing_au", 0.0)
        )

        constant_disk_components[disk_name] = {
            "map": disk_map,
            "amp": disk.get("amp", 1.0),
            "r_in_au": disk.get("r_in_au", 0.0),
            "r_out_au": disk["r_out_au"],
            "edge_smoothing_au": disk.get("edge_smoothing_au", 0.0),
        }

        img_constant_disks_total += disk_map
        img_total += disk_map

    # ============================================================
    # Componente central gaussiana
    # ============================================================
    central_cfg = axisymmetric_disk.get("central_component", {"enabled": False})

    if central_cfg.get("enabled", False):
        img_central = central_gaussian_component(
            r_disk=r_disk,
            amp=central_cfg["amp"],
            sigma_au=central_cfg["sigma_au"]
        )
        img_total += img_central

    # ============================================================
    # Anillos gaussianos
    # ============================================================
    rings = axisymmetric_disk.get("rings", [])

    for i, ring in enumerate(rings):
        if not ring.get("enabled", True):
            continue

        ring_name = ring.get("name", f"ring_{i+1}")

        ring_map = gaussian_ring_component(
            r_disk=r_disk,
            amp=ring["amp"],
            r0_au=ring["r0_au"],
            sigma_au=ring["sigma_au"]
        )

        ring_components[ring_name] = {
            "map": ring_map,
            "amp": ring["amp"],
            "r0_au": ring["r0_au"],
            "sigma_au": ring["sigma_au"],
        }

        img_rings_total += ring_map
        img_total += ring_map

    return {
        "img_total": img_total,

        "img_constant_disks_total": img_constant_disks_total,
        "constant_disk_components": constant_disk_components,

        "img_central": img_central,

        "img_rings_total": img_rings_total,
        "ring_components": ring_components,
    }


# PLANETAS
def disk_polar_to_cartesian(r_au, phi_deg):
    """
    Convierte coordenadas polares del plano del disco a cartesianas.
    """
    phi_rad = np.deg2rad(phi_deg)
    x = r_au * np.cos(phi_rad)
    y = r_au * np.sin(phi_rad)
    return x, y


def gaussian_planet_component(x_disk, y_disk,
                              x0_au, y0_au,
                              amp,
                              sigma_x_au,
                              sigma_y_au,
                              theta_deg=0.0):
    """
    Gaussiana 2D compacta en el plano del disco.
    """

    if sigma_x_au <= 0 or sigma_y_au <= 0:
        raise ValueError("sigma_x_au y sigma_y_au deben ser > 0.")

    dx = x_disk - x0_au
    dy = y_disk - y0_au

    theta_rad = np.deg2rad(theta_deg)
    x_rot =  dx * np.cos(theta_rad) + dy * np.sin(theta_rad)
    y_rot = -dx * np.sin(theta_rad) + dy * np.cos(theta_rad)

    return amp * np.exp(
        -0.5 * (
            (x_rot / sigma_x_au)**2 +
            (y_rot / sigma_y_au)**2
        )
    )


def add_planets(img_base, planets, coords, reference_image=None):
    """
    Añade planetas compactos a una imagen base.
    """

    if reference_image is None:
        reference_image = img_base

    x_disk = coords["x_disk"]
    y_disk = coords["y_disk"]

    img_planets_total = np.zeros_like(img_base)
    planet_components = {}

    global_peak = np.max(reference_image)

    for i, planet in enumerate(planets):
        if not planet.get("enabled", True):
            continue

        planet_name = planet.get("name", f"planet_{i+1}")

        amp_rel = planet["amp_rel"]
        amp_rel_to = planet.get("amp_rel_to", "global_peak")

        r_au = planet["r_au"]
        phi_deg = planet["phi_deg"]
        xp, yp = disk_polar_to_cartesian(r_au, phi_deg)

        sigma_x_au = planet["sigma_r_au"]
        sigma_y_au = planet["sigma_phi_au"]
        theta_deg = planet.get("theta_deg", 0.0)

        if amp_rel_to == "global_peak":
            amp_abs = amp_rel * global_peak

        elif amp_rel_to == "local_disk":
            dist2 = (x_disk - xp)**2 + (y_disk - yp)**2
            iy, ix = np.unravel_index(np.argmin(dist2), dist2.shape)
            local_val = reference_image[iy, ix]
            amp_abs = amp_rel * local_val

        else:
            raise ValueError(
                f"amp_rel_to='{amp_rel_to}' no reconocido. "
                "Usa 'global_peak' o 'local_disk'."
            )

        planet_map = gaussian_planet_component(
            x_disk=x_disk,
            y_disk=y_disk,
            x0_au=xp,
            y0_au=yp,
            amp=amp_abs,
            sigma_x_au=sigma_x_au,
            sigma_y_au=sigma_y_au,
            theta_deg=theta_deg
        )

        planet_components[planet_name] = {
            "map": planet_map,
            "amp_rel": amp_rel,
            "amp_rel_to": amp_rel_to,
            "amp_abs": amp_abs,
            "r_au": r_au,
            "phi_deg": phi_deg,
            "x0_au": xp,
            "y0_au": yp,
            "sigma_x_au": sigma_x_au,
            "sigma_y_au": sigma_y_au,
            "theta_deg": theta_deg,
        }

        img_planets_total += planet_map

    img_total = img_base + img_planets_total

    return {
        "img_total": img_total,
        "img_planets_total": img_planets_total,
        "planet_components": planet_components,
    }


# ASIMETRÍAS AZIMUTALES ANCLADAS A ANILLOS
def wrapped_delta_phi(phi, phi0):
    """
    Diferencia angular mínima entre phi y phi0.

    Ambas cantidades deben estar en radianes.
    El resultado queda en el intervalo (-pi, pi].
    """
    return np.arctan2(np.sin(phi - phi0), np.cos(phi - phi0))


def ring_asymmetry_profile(r_disk, phi_disk,
                           r0_au, sigma_r_au,
                           phi0_deg, sigma_phi_deg,
                           amp_rel):
    """
    Perfil 2D de una asimetría azimutal anclada a un anillo.

    La forma funcional es:

        f(r,phi) = amp_rel
                   * exp[-0.5 * ((r-r0)/sigma_r)^2]
                   * exp[-0.5 * (dphi/sigma_phi)^2]

    donde sigma_phi se interpreta en grados.
    """

    if sigma_r_au <= 0:
        raise ValueError("sigma_r_au debe ser > 0.")

    if sigma_phi_deg <= 0:
        raise ValueError("sigma_phi_deg debe ser > 0.")

    phi0_rad = np.deg2rad(phi0_deg)
    sigma_phi_rad = np.deg2rad(sigma_phi_deg)

    dphi = wrapped_delta_phi(phi_disk, phi0_rad)

    radial_term = np.exp(-0.5 * ((r_disk - r0_au) / sigma_r_au)**2)
    az_term = np.exp(-0.5 * (dphi / sigma_phi_rad)**2)

    return amp_rel * radial_term * az_term


def add_ring_asymmetries(img_base,
                         ring_asymmetries,
                         ring_components,
                         coords):
    """
    Añade asimetrías azimutales ancladas a anillos existentes.

    Parameters
    ----------
    img_base : 2D ndarray
        Imagen sobre la que se aplicarán las asimetrías.
        En esta base ya puede venir el disco + planetas.
    ring_asymmetries : list of dict
        Lista de configuraciones de asimetrías.
    ring_components : dict
        Diccionario con los anillos individuales generado por
        make_axisymmetric_disk().
    coords : dict
        Salida de build_coordinate_grids().

    Returns
    -------
    result : dict
        Contiene:
            img_total
            img_asym_additive_total
            img_asym_multiplicative_effect_total
            asymmetry_components
    """

    r_disk = coords["r_disk"]
    phi_disk = coords["phi_disk"]

    # Empezamos desde la imagen base
    img_total = img_base.copy()

    # Mapas diagnósticos separados
    img_asym_additive_total = np.zeros_like(img_base)
    img_asym_multiplicative_effect_total = np.zeros_like(img_base)

    # Guardamos cada asimetría por separado
    asymmetry_components = {}

    for i, asym in enumerate(ring_asymmetries):
        if not asym.get("enabled", True):
            continue

        asym_name = asym.get("name", f"asym_{i+1}")
        kind = asym.get("kind", "arc")
        ring_name = asym["ring_name"]
        mode = asym.get("mode", "multiplicative").lower()

        if ring_name not in ring_components:
            raise ValueError(
                f"ring_name='{ring_name}' no existe en ring_components."
            )

        # Recuperamos el anillo anfitrión
        host_ring = ring_components[ring_name]
        host_ring_map = host_ring["map"]
        r0_au = host_ring["r0_au"]
        host_sigma_au = host_ring["sigma_au"]

        # Parámetros de la asimetría
        amp_rel = asym["amp_rel"]
        phi0_deg = asym["phi0_deg"]
        sigma_phi_deg = asym["sigma_phi_deg"]
        sigma_r_scale = asym.get("sigma_r_scale", 1.0)

        if sigma_r_scale <= 0:
            raise ValueError("sigma_r_scale debe ser > 0.")

        sigma_r_au = sigma_r_scale * host_sigma_au

        # Perfil relativo 2D de la asimetría
        asym_profile = ring_asymmetry_profile(
            r_disk=r_disk,
            phi_disk=phi_disk,
            r0_au=r0_au,
            sigma_r_au=sigma_r_au,
            phi0_deg=phi0_deg,
            sigma_phi_deg=sigma_phi_deg,
            amp_rel=amp_rel
        )

        # -----------------------------
        # Modo multiplicativo:
        # modifica localmente el anillo anfitrión
        #
        # I_new = I_ring * (1 + f)
        # efecto adicional = I_ring * f
        # -----------------------------
        if mode == "multiplicative":
            asym_effect_map = host_ring_map * asym_profile

            img_total += asym_effect_map
            img_asym_multiplicative_effect_total += asym_effect_map

        # -----------------------------
        # Modo aditivo:
        # suma un componente extra anclado al anillo.
        #
        # Como acordamos amplitudes relativas, usamos el anillo
        # anfitrión como referencia local:
        # I_add = I_ring * f
        #
        # Nota:
        # en esta formulación aditiva sigue estando "anclado" al
        # anillo. No es un arco absoluto independiente del anillo.
        # Eso es consistente con tu planteamiento actual.
        # -----------------------------
        elif mode == "additive":
            asym_effect_map = host_ring_map * asym_profile

            img_total += asym_effect_map
            img_asym_additive_total += asym_effect_map

        else:
            raise ValueError(
                f"mode='{mode}' no reconocido. "
                "Usa 'additive' o 'multiplicative'."
            )

        asymmetry_components[asym_name] = {
            "kind": kind,
            "ring_name": ring_name,
            "mode": mode,
            "amp_rel": amp_rel,
            "phi0_deg": phi0_deg,
            "sigma_phi_deg": sigma_phi_deg,
            "sigma_r_scale": sigma_r_scale,
            "sigma_r_au": sigma_r_au,
            "profile_map": asym_profile,
            "effect_map": asym_effect_map,
        }

    return {
        "img_total": img_total,
        "img_asym_additive_total": img_asym_additive_total,
        "img_asym_multiplicative_effect_total": img_asym_multiplicative_effect_total,
        "asymmetry_components": asymmetry_components,
    }


# MODELO COMPLETO
def build_intrinsic_emission_model(model_config):
    """
    Construye el modelo intrínseco completo:
    - disco axisimétrico
    - planetas compactos
    - asimetrías azimutales ancladas a anillos
    """

    # -----------------------------
    # Geometría
    # -----------------------------
    coords = build_coordinate_grids(model_config["geometry"])

    # -----------------------------
    # Disco axisimétrico base
    # -----------------------------
    axisym_result = make_axisymmetric_disk(
        model_config["axisymmetric_disk"],
        coords
    )

    img_axisym = axisym_result["img_total"]

    # -----------------------------
    # Planetas compactos
    # -----------------------------
    planets = model_config.get("planets", [])

    planets_result = add_planets(
        img_base=img_axisym,
        planets=planets,
        coords=coords,
        reference_image=img_axisym
    )

    img_with_planets = planets_result["img_total"]

    # -----------------------------
    # Asimetrías azimutales
    # -----------------------------
    ring_asymmetries = model_config.get("ring_asymmetries", [])

    asym_result = add_ring_asymmetries(
        img_base=img_with_planets,
        ring_asymmetries=ring_asymmetries,
        ring_components=axisym_result["ring_components"],
        coords=coords
    )

    # -----------------------------
    # Empaquetamos todo
    # -----------------------------
    return {
      "coords": coords,

      # Base axisimétrica
      "img_axisym": img_axisym,
      "img_constant_disks_total": axisym_result["img_constant_disks_total"],
      "constant_disk_components": axisym_result["constant_disk_components"],
      "img_central": axisym_result["img_central"],
      "img_rings_total": axisym_result["img_rings_total"],
      "ring_components": axisym_result["ring_components"],

      # Planetas
      "img_planets_total": planets_result["img_planets_total"],
      "planet_components": planets_result["planet_components"],

      # Asimetrías
      "img_asym_additive_total": asym_result["img_asym_additive_total"],
      "img_asym_multiplicative_effect_total": asym_result["img_asym_multiplicative_effect_total"],
      "asymmetry_components": asym_result["asymmetry_components"],

      # Modelo final
      "img_total": asym_result["img_total"],
    }


# Plot functions
def plot_image(img, coords, title="", unit_label="relative intensity", plane="sky"):
    """
    Grafica una imagen en el plano del cielo o del disco.
    """

    if plane == "sky":
        x = coords["x_1d"]
        y = coords["y_1d"]
        extent = [x[0], x[-1], y[0], y[-1]]
        xlabel = "x_sky [AU]"
        ylabel = "y_sky [AU]"

    elif plane == "disk":
        extent = [
            coords["x_disk"].min(), coords["x_disk"].max(),
            coords["y_disk"].min(), coords["y_disk"].max()
        ]
        xlabel = "x_disk [AU]"
        ylabel = "y_disk [AU]"

    else:
        raise ValueError("plane debe ser 'sky' o 'disk'.")

    plt.figure(figsize=(6, 5))
    im = plt.imshow(img, origin="lower", extent=extent)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar(im, label=unit_label)
    plt.tight_layout()
    plt.savefig(title + '.png')
    plt.show()


def plot_radial_profile(img, coords, nbins=300, title="Radial profile"):
    """
    Calcula y grafica el perfil radial medio de una imagen.
    """
    r_disk = coords["r_disk"]

    r_flat = r_disk.ravel()
    i_flat = img.ravel()

    bins = np.linspace(r_flat.min(), r_flat.max(), nbins + 1)
    r_centers = 0.5 * (bins[:-1] + bins[1:])
    i_mean = np.full(nbins, np.nan)

    for i in range(nbins):
        mask = (r_flat >= bins[i]) & (r_flat < bins[i+1])
        if np.any(mask):
            i_mean[i] = np.mean(i_flat[mask])

    plt.figure(figsize=(6, 4.5))
    plt.plot(r_centers, i_mean, lw=2)
    plt.xlabel("r_disk [AU]")
    plt.ylabel("Mean relative intensity")
    plt.title(title)
    plt.tight_layout()
    plt.show()

################## Flux normalization to total flux
def normalize_to_total_flux(img, total_flux):
    """
    Normaliza una imagen para que su flujo total sea igual al
    valor especificado por el usuario.

    Parameters
    ----------
    img : 2D ndarray
        Imagen a normalizar.
    total_flux : float
        Flujo total deseado después de la normalización.
        Si la imagen va a interpretarse como Jy/pixel,
        entonces total_flux estará en Jy.

    Returns
    -------
    result : dict
        Contiene:
            img_normalized : imagen normalizada
            scale_factor   : factor multiplicativo aplicado
            original_sum   : suma original de la imagen
            final_sum      : suma final de la imagen
    """

    # Validamos que el flujo objetivo sea positivo
    if total_flux <= 0:
        raise ValueError("total_flux debe ser > 0.")

    # Calculamos la suma original de la imagen
    original_sum = np.sum(img)

    # Evitamos división por cero o normalización de una imagen sin señal
    if original_sum <= 0:
        raise ValueError(
            "La suma total de la imagen es <= 0. "
            "No se puede normalizar al flujo total."
        )

    # Calculamos el factor de escala necesario
    scale_factor = total_flux / original_sum

    img_normalized = img * scale_factor

    final_sum = np.sum(img_normalized)

    # Devolvemos todo en un diccionario
    return {
        "img_normalized": img_normalized,
        "scale_factor": scale_factor,
        "original_sum": original_sum,
        "final_sum": final_sum,
    }

####################3 Flux normalization to peak emission
def normalize_to_peak_pixel(img, peak_value):
    """
    Normaliza una imagen para que el píxel de máxima emisión
    tenga el valor especificado por el usuario.

    Parameters
    ----------
    img : 2D ndarray
        Imagen a normalizar.
    peak_value : float
        Valor deseado para el píxel de máxima emisión.
        Si la imagen va a interpretarse como Jy/pixel,
        entonces peak_value estará en Jy/pixel.

    Returns
    -------
    result : dict
        Contiene:
            img_normalized : imagen normalizada
            scale_factor   : factor multiplicativo aplicado
            original_peak  : valor máximo original
            final_peak     : valor máximo final
            peak_index     : índice del píxel máximo original
    """

    # Validamos que el pico objetivo sea positivo
    if peak_value <= 0:
        raise ValueError("peak_value debe ser > 0.")

    # Calculamos el valor máximo original
    original_peak = np.max(img)

    if original_peak <= 0:
        raise ValueError(
            "El pico máximo de la imagen es <= 0. "
            "No se puede normalizar al pico."
        )

    # Obtenemos la posición del píxel de máxima emisión
    peak_index = np.unravel_index(np.argmax(img), img.shape)

    # Calculamos el factor de escala necesario
    scale_factor = peak_value / original_peak
    img_normalized = img * scale_factor

    # Verificamos el pico final
    final_peak = np.max(img_normalized)


    return {
        "img_normalized": img_normalized,
        "scale_factor": scale_factor,
        "original_peak": original_peak,
        "final_peak": final_peak,
        "peak_index": peak_index,
    }


def normalize_model_image(img, method, value):
    """
    Función envolvente para normalizar una imagen usando uno de
    dos métodos:

    - method='total_flux'
    - method='peak_pixel'

    Parameters
    ----------
    img : 2D ndarray
        Imagen a normalizar.
    method : str
        Método de normalización.
    value : float
        Valor objetivo de la normalización.

    Returns
    -------
    result : dict
        Diccionario con la imagen normalizada y metadatos.
    """

    method = method.lower()

    if method == "total_flux":
        result = normalize_to_total_flux(img, total_flux=value)
        result["method"] = "total_flux"
        result["target_value"] = value
        return result

    elif method == "peak_pixel":
        result = normalize_to_peak_pixel(img, peak_value=value)
        result["method"] = "peak_pixel"
        result["target_value"] = value
        return result

    else:
        raise ValueError(
            f"method='{method}' no reconocido. "
            "Usa 'total_flux' o 'peak_pixel'."
        )

# APLICAR NORMALIZACIÓN A TODOS LOS COMPONENTES DEL MODELO
def apply_normalization_to_model_components(model, norm_result):
    """
    Aplica el mismo factor de escala a todos los componentes del
    modelo y genera versiones normalizadas sin modificar las
    originales.

    Parameters
    ----------
    model : dict
        Diccionario generado por build_intrinsic_emission_model().
    norm_result : dict
        Salida de normalize_model_image(), debe contener:
            scale_factor

    Returns
    -------
    model_out : dict
        Copia del modelo con todos los componentes normalizados
        añadidos.
    """

    # Validamos que exista el factor de escala
    if "scale_factor" not in norm_result:
        raise ValueError("norm_result debe contener 'scale_factor'.")

    scale_factor = norm_result["scale_factor"]
    model_out = model.copy()

    model_out["normalization"] = norm_result


    # IMAGEN TOTAL
    model_out["img_total_normalized"] = model["img_total"] * scale_factor

    # COMPONENTES PRINCIPALES (si existen)
    optional_maps = [
        "img_axisym",
        "img_central",
        "img_rings_total",
        "img_planets_total",
        "img_asym_additive_total",
        "img_asym_multiplicative_effect_total",
    ]

    for key in optional_maps:
        if key in model and model[key] is not None:
            model_out[f"{key}_normalized"] = model[key] * scale_factor

    # ANILLOS INDIVIDUALES
    if "ring_components" in model:
        ring_components_norm = {}

        for ring_name, ring_info in model["ring_components"].items():

            ring_map = ring_info["map"]

            # Copiamos toda la info y añadimos mapa normalizado
            ring_info_norm = ring_info.copy()
            ring_info_norm["map_normalized"] = ring_map * scale_factor

            ring_components_norm[ring_name] = ring_info_norm

        model_out["ring_components_normalized"] = ring_components_norm

    # PLANETAS INDIVIDUALES
    if "planet_components" in model:
        planet_components_norm = {}

        for pname, pinfo in model["planet_components"].items():

            p_map = pinfo["map"]

            pinfo_norm = pinfo.copy()
            pinfo_norm["map_normalized"] = p_map * scale_factor

            # También escalamos amplitud absoluta
            if "amp_abs" in pinfo:
                pinfo_norm["amp_abs_normalized"] = pinfo["amp_abs"] * scale_factor

            planet_components_norm[pname] = pinfo_norm

        model_out["planet_components_normalized"] = planet_components_norm

    # ASIMETRÍAS
    if "asymmetry_components" in model:
        asym_components_norm = {}

        for aname, ainfo in model["asymmetry_components"].items():

            ainfo_norm = ainfo.copy()

            # Perfil relativo NO se escala (es adimensional)
            # profile_map se deja intacto

            # effect_map sí se escala
            if "effect_map" in ainfo:
                ainfo_norm["effect_map_normalized"] = (
                    ainfo["effect_map"] * scale_factor
                )

            asym_components_norm[aname] = ainfo_norm

        model_out["asymmetry_components_normalized"] = asym_components_norm

    return model_out

# BEAM, CONVOLUTION AND UNIT TRANSFORM (A JY/BEAM)
from astropy.convolution import Gaussian2DKernel, convolve_fft

def fwhm_to_sigma(fwhm):
    """
    Convierte FWHM a sigma para una gaussiana.

    Parameters
    ----------
    fwhm : float
        Full width at half maximum.

    Returns
    -------
    sigma : float
        Desviación estándar de la gaussiana.
    """
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def beam_area_arcsec2(bmaj_arcsec, bmin_arcsec):
    """
    Calcula el área de un beam gaussiano elíptico en arcsec^2.

    Formula
    -------
    Omega_beam = pi / (4 ln 2) * bmaj * bmin

    Parameters
    ----------
    bmaj_arcsec : float
        Eje mayor FWHM del beam [arcsec].
    bmin_arcsec : float
        Eje menor FWHM del beam [arcsec].

    Returns
    -------
    omega_beam : float
        Área del beam [arcsec^2].
    """

    if bmaj_arcsec <= 0 or bmin_arcsec <= 0:
        raise ValueError("bmaj_arcsec y bmin_arcsec deben ser > 0.")

    omega_beam = (np.pi / (4.0 * np.log(2.0))) * bmaj_arcsec * bmin_arcsec
    return omega_beam


def make_gaussian_beam_kernel(pixscale_au,
                              distance_pc,
                              bmaj_arcsec,
                              bmin_arcsec,
                              bpa_deg):
    """
    Construye un kernel gaussiano elíptico usando astropy.

    Parameters
    ----------
    pixscale_au : float
        Escala del pixel [AU/pixel].
    distance_pc : float
        Distancia a la fuente [pc].
    bmaj_arcsec : float
        Beam major axis FWHM [arcsec].
    bmin_arcsec : float
        Beam minor axis FWHM [arcsec].
    bpa_deg : float
        Beam position angle [deg], convención FITS
        (de Norte hacia Este).

    Returns
    -------
    kernel : astropy.convolution.Gaussian2DKernel
        Kernel gaussiano elíptico normalizado.
    """

    # --------------------------------------------------------
    # Convertimos la escala de pixel a arcsec/pixel
    # --------------------------------------------------------
    pixscale_arcsec = pixscale_au / distance_pc

    if pixscale_arcsec <= 0:
        raise ValueError("pixscale_arcsec debe ser > 0.")

    # --------------------------------------------------------
    # Convertimos FWHM del beam a sigma en arcsec
    # --------------------------------------------------------
    sigma_maj_arcsec = fwhm_to_sigma(bmaj_arcsec)
    sigma_min_arcsec = fwhm_to_sigma(bmin_arcsec)

    # --------------------------------------------------------
    # Convertimos sigma de arcsec a pixeles
    # --------------------------------------------------------
    sigma_maj_pix = sigma_maj_arcsec / pixscale_arcsec
    sigma_min_pix = sigma_min_arcsec / pixscale_arcsec

    # --------------------------------------------------------
    # Conversión angular:
    # BPA en FITS está medido desde +y hacia +x
    # theta en Gaussian2DKernel está medido desde +x
    # --------------------------------------------------------
    theta_rad = np.deg2rad(90.0 - bpa_deg)

    # --------------------------------------------------------
    # Construimos el kernel.
    # x_stddev corresponde al eje x del array.
    # y_stddev corresponde al eje y del array.
    # --------------------------------------------------------
    kernel = Gaussian2DKernel(
        x_stddev=sigma_min_pix,
        y_stddev=sigma_maj_pix,
        theta=theta_rad
    )

    return kernel


def convolve_model_with_beam(img_jypix,
                             pixscale_au,
                             distance_pc,
                             bmaj_arcsec,
                             bmin_arcsec,
                             bpa_deg,
                             return_kernel=False):
    """
    Convoluciona una imagen en Jy/pixel con un beam gaussiano.

    La salida permanece en Jy/pixel.

    Parameters
    ----------
    img_jypix : 2D ndarray
        Imagen de entrada en Jy/pixel.
    pixscale_au : float
        Escala del pixel [AU/pixel].
    distance_pc : float
        Distancia a la fuente [pc].
    bmaj_arcsec, bmin_arcsec : float
        FWHM del beam [arcsec].
    bpa_deg : float
        Beam position angle [deg].
    return_kernel : bool
        Si es True, devuelve también el kernel.

    Returns
    -------
    result : dict
        Contiene:
            img_convolved_jypix
            kernel              (opcional)
    """

    # Kernel
    kernel = make_gaussian_beam_kernel(
        pixscale_au=pixscale_au,
        distance_pc=distance_pc,
        bmaj_arcsec=bmaj_arcsec,
        bmin_arcsec=bmin_arcsec,
        bpa_deg=bpa_deg
    )

    # Aplicamos la convolución FFT
    img_convolved_jypix = convolve_fft(
        img_jypix,
        kernel,
        boundary="fill",
        fill_value=0.0,
        normalize_kernel=False
    )

    result = {
        "img_convolved_jypix": img_convolved_jypix
    }

    if return_kernel:
        result["kernel"] = kernel

    return result


def convert_jypixel_to_jybeam(img_jypix,
                              pixscale_au,
                              distance_pc,
                              bmaj_arcsec,
                              bmin_arcsec):
    """
    Convierte una imagen de Jy/pixel a Jy/beam.

    Parameters
    ----------
    img_jypix : 2D ndarray
        Imagen en Jy/pixel.
    pixscale_au : float
        Escala del pixel [AU/pixel].
    distance_pc : float
        Distancia a la fuente [pc].
    bmaj_arcsec, bmin_arcsec : float
        FWHM del beam [arcsec].

    Returns
    -------
    result : dict
        Contiene:
            img_jybeam
            conversion_factor
            omega_pix_arcsec2
            omega_beam_arcsec2
    """

    # Escala angular del pixel en arcsec/pixel
    pixscale_arcsec = pixscale_au / distance_pc

    if pixscale_arcsec <= 0:
        raise ValueError("pixscale_arcsec debe ser > 0.")

    # area angular del pixel
    omega_pix_arcsec2 = pixscale_arcsec**2

    # area angular del beam
    omega_beam_arcsec2 = beam_area_arcsec2(bmaj_arcsec, bmin_arcsec)

    # Jy/pixel -> Jy/beam
    conversion_factor = omega_beam_arcsec2 / omega_pix_arcsec2

    img_jybeam = img_jypix * conversion_factor

    return {
        "img_jybeam": img_jybeam,
        "conversion_factor": conversion_factor,
        "omega_pix_arcsec2": omega_pix_arcsec2,
        "omega_beam_arcsec2": omega_beam_arcsec2,
    }


def add_gaussian_noise_jybeam(img_jybeam,
                              noise_std_jybeam,
                              random_seed=None):
    """
    Agrega ruido gaussiano en unidades de Jy/beam.

    Parameters
    ----------
    img_jybeam : 2D ndarray
        Imagen de entrada en Jy/beam.
    noise_std_jybeam : float
        Desviación estándar del ruido [Jy/beam].
    random_seed : int or None
        Semilla aleatoria para reproducibilidad.

    Returns
    -------
    result : dict
        Contiene:
            img_noisy_jybeam
            noise_map_jybeam
            noise_std_jybeam
    """

    if noise_std_jybeam < 0:
        raise ValueError("noise_std_jybeam debe ser >= 0.")

    rng = np.random.default_rng(random_seed)

    noise_map_jybeam = rng.normal(
        loc=0.0,
        scale=noise_std_jybeam,
        size=img_jybeam.shape
    )

    img_noisy_jybeam = img_jybeam + noise_map_jybeam

    return {
        "img_noisy_jybeam": img_noisy_jybeam,
        "noise_map_jybeam": noise_map_jybeam,
        "noise_std_jybeam": noise_std_jybeam,
    }


def convolve_convert_and_add_noise(img_jypix,
                                   pixscale_au,
                                   distance_pc,
                                   bmaj_arcsec,
                                   bmin_arcsec,
                                   bpa_deg,
                                   noise_std_jybeam=0.0,
                                   random_seed=None,
                                   return_kernel=False):
    """
    Pipeline completo:
    1) convoluciona imagen en Jy/pixel con beam
    2) convierte a Jy/beam
    3) agrega ruido gaussiano en Jy/beam

    Parameters
    ----------
    img_jypix : 2D ndarray
        Imagen de entrada en Jy/pixel.
    pixscale_au : float
        Escala del pixel [AU/pixel].
    distance_pc : float
        Distancia a la fuente [pc].
    bmaj_arcsec, bmin_arcsec : float
        FWHM del beam [arcsec].
    bpa_deg : float
        Beam position angle [deg].
    noise_std_jybeam : float
        Ruido rms [Jy/beam].
    random_seed : int or None
        Semilla aleatoria.
    return_kernel : bool
        Si True, también devuelve el kernel.

    Returns
    -------
    result : dict
        Contiene:
            img_convolved_jypix
            img_convolved_jybeam
            img_noisy_jybeam
            noise_map_jybeam
            conversion_factor
            omega_pix_arcsec2
            omega_beam_arcsec2
            kernel (opcional)
    """

    # convolución con beam
    conv_result = convolve_model_with_beam(
        img_jypix=img_jypix,
        pixscale_au=pixscale_au,
        distance_pc=distance_pc,
        bmaj_arcsec=bmaj_arcsec,
        bmin_arcsec=bmin_arcsec,
        bpa_deg=bpa_deg,
        return_kernel=return_kernel
    )

    img_convolved_jypix = conv_result["img_convolved_jypix"]

    #conversión a Jy/beam

    conv_units_result = convert_jypixel_to_jybeam(
        img_jypix=img_convolved_jypix,
        pixscale_au=pixscale_au,
        distance_pc=distance_pc,
        bmaj_arcsec=bmaj_arcsec,
        bmin_arcsec=bmin_arcsec
    )

    img_convolved_jybeam = conv_units_result["img_jybeam"]

    # agregar ruido en Jy/beam
    noise_result = add_gaussian_noise_jybeam(
        img_jybeam=img_convolved_jybeam,
        noise_std_jybeam=noise_std_jybeam,
        random_seed=random_seed
    )

    result = {
        "img_convolved_jypix": img_convolved_jypix,
        "img_convolved_jybeam": img_convolved_jybeam,
        "img_noisy_jybeam": noise_result["img_noisy_jybeam"],
        "noise_map_jybeam": noise_result["noise_map_jybeam"],
        "noise_std_jybeam": noise_result["noise_std_jybeam"],
        "conversion_factor": conv_units_result["conversion_factor"],
        "omega_pix_arcsec2": conv_units_result["omega_pix_arcsec2"],
        "omega_beam_arcsec2": conv_units_result["omega_beam_arcsec2"],
    }

    if return_kernel:
        result["kernel"] = conv_result["kernel"]

    return result


def comparar_graficos(data1, data2, titulo1="Gráfico 1", titulo2="Gráfico 2"):
    """
    Crea una figura con dos subplots para comparar dos conjuntos de datos.
    """
    # Creamos la figura y los ejes (1 fila, 2 columnas)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Primer gráfico
    ax1.imshow(data1, label=titulo1)
    ax1.set_title(titulo1)
    #ax1.set_xlabel(etiqueta_x)

    # Segundo gráfico
    ax2.imshow(data2, label=titulo2)
    ax2.set_title(titulo2)
    #ax2.set_xlabel(etiqueta_x)

    # Ajustar el diseño para que no se traslapen los títulos
    plt.tight_layout()
    plt.show()


# to normalize using the spected noise
def normalize_to_peak_snr(img, rms_noise, target_snr=100.0):
    """
    Normaliza una imagen para que su píxel de máxima emisión tenga
    una razón señal-a-ruido especificada.

    Parameters
    ----------
    img : 2D ndarray
        Imagen sin ruido. Debe estar en las mismas unidades que rms_noise,
        por ejemplo Jy/beam.
    rms_noise : float
        Ruido rms que se agregará después. Debe estar en las mismas
        unidades que img.
    target_snr : float
        SNR deseado para el píxel de máxima emisión.

    Returns
    -------
    result : dict
        Contiene:
            img_normalized
            scale_factor
            original_peak
            target_peak
            final_peak
            rms_noise
            target_snr
            final_snr_peak
    """

    if rms_noise <= 0:
        raise ValueError("rms_noise debe ser > 0.")

    if target_snr <= 0:
        raise ValueError("target_snr debe ser > 0.")

    original_peak = np.max(img)

    if original_peak <= 0:
        raise ValueError(
            "El pico de la imagen es <= 0. "
            "No se puede normalizar a un SNR positivo."
        )

    target_peak = target_snr * rms_noise
    scale_factor = target_peak / original_peak

    img_normalized = img * scale_factor

    final_peak = np.max(img_normalized)
    final_snr_peak = final_peak / rms_noise

    return {
        "img_normalized": img_normalized,
        "scale_factor": scale_factor,
        "original_peak": original_peak,
        "target_peak": target_peak,
        "final_peak": final_peak,
        "rms_noise": rms_noise,
        "target_snr": target_snr,
        "final_snr_peak": final_snr_peak,
    }