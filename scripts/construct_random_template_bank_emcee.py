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
parser.add_argument('-o', '--output_path', default="generated_template_banks/")
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
parser.add_argument('-z', '--templates', default='100', type=int)
parser.add_argument('-file', '--filename', default='3D_template_bank', type=str)
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
output_path = args.output_path
os.makedirs(output_path, exist_ok=True)
low_omega = 2 * np.pi / p_orb_high
high_omega = 2 * np.pi / p_orb_low

ndim, nwalkers = 3, 128 #jjc
pre_burnin = 5000
#np.random.seed(42)
filename = output_path + args.filename + '.h5'
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
    mcmc_steps = math.ceil(args.templates / nwalkers * thin * 1.1) - pre_burnin
    #sampler.reset()
    start = time.time()
    sampler.run_mcmc(state, mcmc_steps, progress=True)
    logger.info("Main Phase took {:.1f} seconds".format(time.time() - start))

sampler = emcee.backends.HDFBackend(filename, read_only=True)
tau = sampler.get_autocorr_time()
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))
logger.info(f"Using burn-in of {burnin} and thinning of {thin} based on autocorrelation time.")
#template_bank = sampler.get_chain(discard=burnin, flat=True, thin=thin)
final_template_bank = sampler.get_chain(discard=burnin, flat=True, thin=thin)
#indices = np.random.choice(len(template_bank), args.templates, replace=False)
#final_template_bank = template_bank[indices]

# Dump template bank to file
with open(output_path + args.filename + '_peasoup_format.txt', 'w') as f:
    f.write(f'# TEMPLATE BANK TECHNIQUE: RANDOM\n')
    f.write(f'# TOBS (h): {args.obs_time/60}\n')
    f.write(f'# COVERAGE: {coverage}\n')
    f.write(f'# MISMATCH: {mismatch}\n')
    f.write(f'# SPIN PERIOD (ms): {args.spin_period}\n')
    f.write(f'# PORB MIN (h): {args.porb_low/60}\n')
    f.write(f'# PORB MAX (h): {args.porb_high/60}\n')
    f.write(f'# MIN PULSAR MASS (Msun): {min_pulsar_mass}\n')
    f.write(f'# MAX COMPANION MASS (Msun): {max_companion_mass}\n')
    f.write(f'# INCLINATION ANGLE FRACTION: {alpha}\n')
    f.write(f'# ORBITAL PHASE MIN (rad): 0\n')
    f.write(f'# ORBITAL PHASE MAX (rad): {max_phase}\n')
    f.write(f'# NTEMPLATES: {args.templates + 1}\n')  # Add 1 for the asini/c=0 template
    f.write('# ANGULAR_VELOCITY (rad/s) ASINI/C (lt-s) ORB_PHASE (rad)\n')
    for tpl in final_template_bank:
        f.write(f'{tpl[0]} {tpl[1]} {tpl[2]}\n')

    # Add the extra template: same omega and phase as a random template, but asini/c = 0 for sensitivity towards isolated pulsars.
    random_tpl = random.choice(final_template_bank)
    omega, _, orb_phase = random_tpl
    f.write(f'{omega} 0.0 {orb_phase}\n')

# Generate corner plot
try:
    tau = sampler.get_autocorr_time()
    burnin = int(2 * np.max(tau))
    thin = int(0.5 * np.min(tau))
    samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
    log_probs = np.log10(np.exp(sampler.get_log_prob(discard=burnin, flat=True, thin=thin)))
    logger.info(f"Using burn-in of {burnin} and thinning of {thin} based on autocorrelation time.")
except Exception as e:
    logger.warning(f"Autocorrelation time could not be estimated: {e}")
    logger.warning("Proceeding without burn-in or thinning for corner plot.")
    samples = sampler.get_chain(flat=True)
    log_probs = np.log10(np.exp(sampler.get_log_prob(flat=True)))

samples[:, 0] = (2 * np.pi / samples[:, 0]) / 3600
samples[:, 2] = np.degrees(samples[:, 2])
all_samples = np.concatenate((samples, log_probs[:, None]), axis=1)
#labels = ["Orbital Period \n (hrs)", "Projected Radius \n (lt-s)", "Orbital Phase \n (degrees)", r"log$_{10}$ $\\left(\\sqrt{|det(\\gamma_{\\alpha \\beta})|}\\right)$"]
labels = ["Orbital Period \n (hrs)", "Projected Radius \n (lt-s)", "Orbital Phase \n (degrees)", r"$\log_{10} \left(\sqrt{|\det(\gamma_{\alpha \beta})|}\right)$"]

figure = corner.corner(all_samples, labels=labels, color='black', title_kwargs={"fontsize": 12},
                       smooth=True, smooth1d=True, scale_hist=True,
                       levels=(0.1175031, 0.39346934, 0.67534753, 0.86466472))
#figure.savefig(output_path + args.filename + '_corner.pdf')
figure.savefig(output_path + args.filename + '_corner.png')