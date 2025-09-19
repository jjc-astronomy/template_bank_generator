#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Author      : Vishnu Balakrishnan
# Created     : 11 April 2025
# License     : GNU General Public License v3.0
# =============================================================================
# Description :
# This code generates a random template bank for a user-defined number of templates. Run this script after calculating how many templates you need from calculate_required_random_templates_circular_orbit_searches.py
# =============================================================================

import numpy as np
import sympy as sy
sy.init_printing(use_unicode=True)
import sys, time, math, os, argparse, logging
import emcee
from multiprocessing import Pool
from schwimmbad import MPIPool
import multiprocessing, corner
import random

# Argument parser
parser = argparse.ArgumentParser(description='Generate a template-bank for a user-defined no. of templates for coherent full keplerian circular orbit search')
parser.add_argument('-t', '--obs_time', default='72', type=float)
parser.add_argument('-p', '--porb_low', default='360', type=float)
parser.add_argument('-P', '--porb_high', type=float)
parser.add_argument('-c', '--max_comp_mass', default='8', type=float)
parser.add_argument('-d', '--min_pulsar_mass', default='1.4', type=float)
parser.add_argument('-s', '--spin_period', default='5', type=float)
parser.add_argument('-f', '--fraction', default='1', type=float)
parser.add_argument('-b', '--coverage', default='0.9', type=float)
parser.add_argument('-m', '--mismatch', default='0.2', type=float)
parser.add_argument('-n', '--ncpus', default='32', type=int)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--quiet', action='store_true')

args = parser.parse_args()
args.porb_high = args.porb_high if args.porb_high else args.obs_time * 10

# Setup logging
log_level = logging.INFO
if args.debug:
    log_level = logging.DEBUG
elif args.quiet:
    log_level = logging.ERROR
elif args.verbose:
    log_level = logging.INFO

logging.basicConfig(format='[%(levelname)s] %(message)s', level=log_level)
logger = logging.getLogger(__name__)

# Symbolic computation
f, tau, omega, psi, phi, t, T, a, pi, f0 = sy.symbols('f \\tau \\Omega \\psi \\phi t T a \\pi f_0')
phi = 2 * pi * f * (t + tau * sy.sin(omega * t - psi))

def time_average(a):
    return (1/T) * sy.integrate(a, (t, 0, T))

variables = [f, tau, omega, psi]
metric_tensor = np.empty((4, 4), dtype=object)
for i in range(len(variables)):
    for j in range(len(variables)):
        metric_tensor[i][j] = (time_average(sy.diff(phi, variables[i]) * sy.diff(phi, variables[j])) - time_average(sy.diff(phi, variables[i])) * time_average(sy.diff(phi, variables[j])))

metric_tensor3D = np.empty((3, 3), dtype=object)
metric_tensor_w_f_row_column = metric_tensor[1:4, 1:4]
variables = [tau, omega, psi]
for i in range(len(variables)):
    for j in range(len(variables)):
        metric_tensor3D[i][j] = metric_tensor_w_f_row_column[i][j] * metric_tensor[0][0] - (metric_tensor[0][i+1] * metric_tensor[j+1][0])

metric_tensor3D = sy.Matrix(metric_tensor3D)
A = sy.Matrix(3, 3, sy.symbols('A:3:3'))
det_metric_tensor3D = A.det().subs(zip(list(A), list(metric_tensor3D)))
det_metric_tensor3D /= metric_tensor[0][0]**3
expr = det_metric_tensor3D**0.5
expr_numpy = sy.lambdify([f, psi, omega, tau, T, pi], expr, "numpy")

def det_sq_root(angular_velocity, projected_radius, orbital_phase, freq, obs_time):
    return expr_numpy(freq, orbital_phase, angular_velocity, projected_radius, obs_time, np.pi)

def calculate_alpha(sini, max_companion_mass, min_pulsar_mass, current_candidate_mass, current_companion_mass):
    alpha = sini * max_companion_mass * ((current_candidate_mass + current_companion_mass)**(2/3))/(current_companion_mass * (max_companion_mass + min_pulsar_mass)**(2/3))
    return 1 - np.sqrt(1 - alpha**2)

def number_templates(dimension, coverage, mismatch, volume):
    ball_volume = math.pow(np.pi, dimension/2)/math.gamma((dimension/2) + 1)
    return math.log(1 - coverage)/math.log(1 - math.pow(mismatch, dimension/2) * ball_volume/volume)

def log_posterior(theta, freq, obs_time, low_omega, high_omega, max_phase):
    omega, radius, phase = theta
    if not (low_omega < omega < high_omega and 0. < phase < max_phase):
        return -np.inf
    max_radius = alpha * G**(1/3) * max_companion_mass * M_0 * omega**(-2/3)/(c * ((min_pulsar_mass + max_companion_mass) * M_0)**(2/3))
    if not (0. < radius <= max_radius):
        return -np.inf
    det = expr_numpy(freq, phase, omega, radius, obs_time, np.pi)
    if det == 0 or math.isnan(det):
        return -np.inf
    return np.log(det)

# Constants and priors
G, M_0, c = 6.67e-11, 1.989e+30, 2.99792458e+08
obs_time = args.obs_time * 60
p_orb_high = args.porb_high * 60
p_orb_low = args.porb_low * 60
min_pulsar_mass, max_companion_mass, alpha = args.min_pulsar_mass, args.max_comp_mass, args.fraction
coverage, mismatch, ncpus = args.coverage, args.mismatch, args.ncpus
spin_freq = 1/(args.spin_period * 1e-03)
max_phase = 2 * np.pi
low_omega = 2 * np.pi / p_orb_high
high_omega = 2 * np.pi / p_orb_low

ndim, nwalkers = 3, 1500 #jjc
pre_burnin = 5000
#np.random.seed(42)
filename = 'estimate.h5'
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)

initial_guess = np.column_stack((
    np.random.uniform(low_omega, high_omega, nwalkers),
    np.random.uniform(0., alpha * G**(1/3) * max_companion_mass * M_0 * high_omega**(-2/3)/(c * ((min_pulsar_mass + max_companion_mass) * M_0)**(2/3)), nwalkers),
    np.random.uniform(0., max_phase, nwalkers)
))

with Pool(processes=ncpus) as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, backend=backend,
        args=[spin_freq, obs_time, low_omega, high_omega, max_phase], pool=pool)
    start = time.time()
    state = sampler.run_mcmc(initial_guess, pre_burnin, progress=True)
    logger.info("Burn-In Phase took {:.1f} seconds".format(time.time() - start))
    tau = sampler.get_autocorr_time()
    burnin = int(2 * np.max(tau))
    thin = int(0.5 * np.min(tau))
    logger.info(f"Using burn-in of {burnin} and thinning of {thin} based on autocorrelation time.")