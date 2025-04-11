import numpy as np
from numba import njit
from typing import Tuple
import logging
import time

@njit
def calculate_delta_phase_model(omega1, tau1, psi1, t, omega2, tau2, psi2, f1, f2):
    phase1 = 2 * np.pi * f1 * (t + tau1 * np.sin(omega1 * t + psi1))
    phase2 = 2 * np.pi * f2 * (t + tau2 * np.sin(omega2 * t + psi2))
    return phase1 - phase2

@njit
def integrate_detection_statistic(omega1, tau1, psi1, omega2, tau2, psi2, f1, f2, tobs, nmc):
    t = np.linspace(0, tobs, nmc)
    delta_phi = calculate_delta_phase_model(omega1, tau1, psi1, t, omega2, tau2, psi2, f1, f2)
    sin_int = np.sum(np.sin(delta_phi)) * (tobs / nmc)
    cos_int = np.sum(np.cos(delta_phi)) * (tobs / nmc)
    return sin_int ** 2 + cos_int ** 2

def compare_templates_with_fsearch(template1: Tuple[float, float, float],
                                   template2: Tuple[float, float, float],
                                   freq: float, tobs: float, T2: float, T_inv: float, nmc: int,
                                   f_search_range: float, coarse_threshold: float,
                                   coarse_step: float, fine_step: float,
                                   logger: logging.Logger = None) -> Tuple[float, float]:

    omega1, tau1, psi1 = template1
    omega2, tau2, psi2 = template2

    freq_inj = T_inv * np.floor(freq * tobs + 0.5)
    f_bin_min = int(np.floor((freq_inj - f_search_range) * tobs))
    f_bin_max = int(np.ceil((freq_inj + f_search_range) * tobs))

    max_detstat = 0.0
    freq_max = 0.0
    delta_f = coarse_step
    stepped = False

    f_bin = f_bin_min
    while f_bin <= f_bin_max:
        f_search = f_bin * T_inv
        detstat = integrate_detection_statistic(omega1, tau1, psi1, omega2, tau2, psi2, freq, f_search, tobs, nmc)

        if detstat > coarse_threshold * T2 and not stepped:
            if logger:
                logger.debug(f"Refining search around {f_search:.6f} Hz for template pair {template1} and {template2}")
            f_bin -= 2 * delta_f
            delta_f = fine_step
            stepped = True

        if detstat > max_detstat:
            max_detstat = detstat
            freq_max = f_search

        f_bin += delta_f

    mismatch = 1.0 - max_detstat / T2
    if logger:
        logger.debug(f"Max detection statistic for templates {template1} and {template2} at frequency {freq_max:.6f} Hz: {max_detstat:.6f}, mismatch: {mismatch:.6f}")
    return max_detstat, mismatch

def setup_logger(verbose: bool, log_interval: int):
    logger = logging.getLogger("TemplateMatcher")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.handlers = [handler]
    return logger

def log_progress(logger, start_time, interval, current_idx, total, accepted, rejected):
    if time.time() - start_time >= interval:
        logger.info(f"Progress: {current_idx}/{total} processed, Accepted: {accepted}, Rejected: {rejected}")
        return time.time()
    return start_time
