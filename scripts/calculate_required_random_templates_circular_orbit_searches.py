#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Author      : Vishnu Balakrishnan
# Created     : 4 June 2018
# Updated     : 11 April 2025
# License     : GNU General Public License v3.0
# =============================================================================

"""
Description :
This Module has been built to Calculate and Generate Random Templates based on Messenger et al. 2008 (arXiv:0809.5223).
These templates are used to do a fully coherent search for circular binary orbits in radio observations.

      Receipe to Generate Random Template Bank!

1. Initialise your signal phase model and calculate the metric tensor of your parameter space.

2. Compute determinant of this metric. This will be used later as a constant density function to distribute templates
   in your parameter space.

2. Compute proper volume/volume integral of your parameter space and calculate number of required templates
   based on required coverage and mismatch.

4. For each proposal template, draw random values from angular velocity,
   projected radius and orbital phase (parameters of interest).

5. Implement a MCMC based on metropolis hastings algorithm using square root of the determinant of the metric tensor as your constant density function. Write Results to file.

6. Step 5 is optional and is only triggered if the user specifies a filename to write the results to. My recommendation for generating large template-banks is to first get the total templates you need from this code and pass that to the emcee version construct_random_template_bank_emcee.py

"""

import sys
import numpy as np
import math
import os
import sympy as sy
import argparse
from multiprocessing import Pool

parser = argparse.ArgumentParser(description='Calculate required templates for a coherent keplerian circular orbit search based on Messenger at al. 2008 and generate template-bank based on Metropolis hastings')
parser.add_argument('-o', '--output_path', help='Output path to save results',  default="generated_template_banks/")
parser.add_argument('-t', '--obs_time', help='Observation time in minutes', default='72', type=float)
parser.add_argument('-p', '--porb_low', help='Lower limit of Orbital Period in minutes', default='360', type=float)
parser.add_argument('-P', '--porb_high', help='Upper limit of Orbital Period in minutes', type=float)
parser.add_argument('-c', '--max_comp_mass', help='Maximum mass of Companion in solar mass units', default='8', type=float)
parser.add_argument('-d', '--min_pulsar_mass', help='Minimum mass of Pulsar in solar mass units', default='1.4', type=float)
parser.add_argument('-s', '--spin_period', help='Fastest spin period of pulsar in ms', default='5', type=float)
parser.add_argument('-f', '--fraction', help='Probability fraction of orbits of different inclination angles', default='1', type=float)
parser.add_argument('-b', '--coverage', help='Coverage of template-bank', default='0.9', type=float)
parser.add_argument('-m', '--mismatch', help='Mismatch of template-bank', default='0.2', type=float)
parser.add_argument('-n', '--ncpus', help='Number of CPUs to use for calculation', default='32', type=int)
parser.add_argument('-i', '--nmc', help='Number of iterations for monte-carlo integration', default='100000', type=int)
parser.add_argument('-file', '--output_filename', help='Filename for template-bank', type=str)

args = parser.parse_args()
args.porb_high = args.porb_high if args.porb_high else args.obs_time * 10

# Define all variable for metric tensor calculations
output_path = args.output_path
f, tau, omega, psi, phi, t, T, a, pi, f0 = sy.symbols('f \tau \Omega \psi \\phi t T a \\pi f_0')
sy.init_printing(use_unicode=True) # pretty printing

# Phase Model for Circular Binary Orbits
phi = 2 * pi * f * (t + tau * sy.sin(omega * t + psi))

def time_average(a):
    b = (1/T) * sy.integrate(a, (t, 0, T))
    return b

variables = [f, tau, omega, psi]

metric_tensor = np.empty(shape=(4, 4), dtype=object)
for i in range(len(variables)):
    for j in range(len(variables)):
        metric_tensor[i][j] = (
            time_average(sy.diff(phi, variables[i]) * sy.diff(phi, variables[j]))
            - time_average(sy.diff(phi, variables[i])) * time_average(sy.diff(phi, variables[j]))
        )

metric_tensor_w_f_row_column = metric_tensor[1:4, 1:4]
variables = [tau, omega, psi]
metric_tensor3D = np.empty(shape=(3, 3), dtype=object)
for i in range(len(variables)):
    for j in range(len(variables)):
        metric_tensor3D[i][j] = metric_tensor_w_f_row_column[i][j] * metric_tensor[0][0] - (
            metric_tensor[0][i + 1] * metric_tensor[j + 1][0])
metric_tensor3D = sy.Matrix(metric_tensor3D)

# manual determinant for python3 compatibility
A = sy.Matrix(3, 3, sy.symbols('A:3:3'))
det_metric_tensor3D = A.det().subs(zip(list(A), list(metric_tensor3D)))
det_metric_tensor3D = det_metric_tensor3D / metric_tensor[0][0]**3
expr = det_metric_tensor3D**0.5
expr_numpy = sy.lambdify([f, psi, omega, tau, T, pi], expr, "numpy")

def det_sq_root(angular_velocity, projected_radius, orbital_phase, freq, obs_time):
    return expr_numpy(freq, orbital_phase, angular_velocity, projected_radius, obs_time, np.pi)

def calculate_alpha(sini, max_companion_mass, min_pulsar_mass, current_pulsar_mass, current_companion_mass):
    alpha = sini * max_companion_mass * ((current_pulsar_mass + current_companion_mass)**(2/3))/(current_companion_mass * (max_companion_mass + min_pulsar_mass)**(2/3))
    p = 1 - np.sqrt(1 - alpha**2)
    return p

def number_templates(dimension, coverage, mismatch, volume):
    n_dim_ball_volume = math.pow(np.pi, dimension/2) / math.gamma((dimension/2) + 1)
    N = math.log(1 - coverage) / math.log(1 - math.pow(mismatch, dimension/2) * n_dim_ball_volume / volume)
    return N

G = 6.67e-11
M_0 = 1.989e+30
c = 2.99792458e+08
pi_1 = np.pi
obs_time = args.obs_time * 60
p_orb_upper_limit = args.porb_high * 60
p_orb_low_limit = args.porb_low * 60
min_pulsar_mass = args.min_pulsar_mass
max_companion_mass = args.max_comp_mass
alpha = args.fraction
volume_integral_iterations = args.nmc
ncpus = args.ncpus
coverage = args.coverage
mismatch = args.mismatch
fastest_spin_period_ms = args.spin_period
freq = 1 / (fastest_spin_period_ms * 1e-03)
max_initial_orbital_phase = 2 * np.pi
probability = calculate_alpha(alpha, max_companion_mass, min_pulsar_mass, min_pulsar_mass, max_companion_mass)
highest_angular_velocity = 2 * np.pi / p_orb_upper_limit
highest_limit_projected_radius = alpha * G**(1/3) * max_companion_mass * M_0 * highest_angular_velocity**(-2/3)/(c * ((min_pulsar_mass + max_companion_mass) * M_0)**(2/3))

def volume_integral(angular_velocity, projected_radius, orbital_phase):
    max_projected_radius = alpha * G**(1/3) * max_companion_mass * M_0 * angular_velocity**(-2/3)/(c * ((min_pulsar_mass + max_companion_mass) * M_0)**(2/3))
    if projected_radius <= max_projected_radius:
        return expr_numpy(freq, orbital_phase, angular_velocity, projected_radius, obs_time, np.pi)
    return 0

samples = np.random.uniform(low=[(2 * np.pi)/p_orb_upper_limit, 0., 0.],
                            high=[(2 * np.pi)/p_orb_low_limit, highest_limit_projected_radius, max_initial_orbital_phase],
                            size=(volume_integral_iterations, 3))

with Pool(processes=ncpus) as pool:
    vals = pool.starmap(volume_integral, samples)
vals = np.array(vals)

volume_element = ((2 * np.pi)/p_orb_low_limit - (2 * np.pi)/p_orb_upper_limit) * highest_limit_projected_radius * max_initial_orbital_phase
volume_integral_result = volume_element * np.mean(vals)
estimated_volume_integral_error = volume_element * np.std(vals) / np.sqrt(volume_integral_iterations)

print('Volume Integral: ', volume_integral_result, 'Volume Integral Error: ', estimated_volume_integral_error)
print('Volume integral error is: %.2f' % ((estimated_volume_integral_error/volume_integral_result) * 100), ' %')
total_templates_targed_search = number_templates(3, coverage, mismatch, np.around(volume_integral_result))

print('observation time (mins):', obs_time/60, 'mass companion:', max_companion_mass, 'orbital period low (hrs):', p_orb_low_limit/3600, 'orbital period high (hrs):', p_orb_upper_limit/3600, 'spin period (ms):', (1/freq) * 1e+3, 'prob:', probability, 'templates: ', total_templates_targed_search, 'integration error percentage: ', (estimated_volume_integral_error/volume_integral_result) * 100, 'coverage: ', coverage, 'mismatch: ', mismatch, 'phase: ', max_initial_orbital_phase)

if not args.output_filename:
    sys.exit()

burn_in_steps = 100
output_filename = args.output_filename
N = int(np.around(total_templates_targed_search + burn_in_steps))
angular_velocity_init = np.random.uniform((2 * np.pi)/p_orb_upper_limit, (2 * np.pi)/p_orb_low_limit)
max_projected_radius = alpha * G**(1/3) * max_companion_mass * M_0 * angular_velocity_init**(-2/3)/(c * ((min_pulsar_mass + max_companion_mass) * M_0)**(2/3))
projected_radius_init = np.random.uniform(0, max_projected_radius)
orbital_phase_init = np.random.uniform(0, 2*np.pi)

p = det_sq_root(angular_velocity_init, projected_radius_init, orbital_phase_init, freq, obs_time)
counts = 0
ntrials = 0
accepted_templates = []
rejected_templates = []
while counts != N:
    angular_velocity = np.random.uniform((2 * np.pi)/p_orb_upper_limit, (2 * np.pi)/p_orb_low_limit)
    max_projected_radius = alpha * G**(1/3) * max_companion_mass * M_0 * angular_velocity**(-2/3)/(c * ((min_pulsar_mass + max_companion_mass) * M_0)**(2/3))
    projected_radius = np.random.uniform(0, max_projected_radius)
    orbital_phase = np.random.uniform(0, 2*np.pi)

    p_prop = det_sq_root(angular_velocity, projected_radius, orbital_phase, freq, obs_time)
    u = np.random.rand()
    if u < p_prop/p:
        p = p_prop
        counts += 1
        accepted_templates.append([angular_velocity, projected_radius, orbital_phase, p])

    ntrials += 1

accepted_templates = np.asarray(accepted_templates)
accepted_templates = accepted_templates[burn_in_steps:]

if not os.path.exists(output_path + output_filename + '_circular_orbit_header.txt'):
    with open(output_path + output_filename + '_circular_orbit_header.txt', 'w') as outfile:
        outfile.write('observation time (mins): ' + str(obs_time/60) + ',' + 'orbital period low (hrs): ' + str(p_orb_low_limit/3600) + ',' + 'orbital period high (hrs): ' + str(p_orb_upper_limit/3600) + ',' + 'spin period (ms): ' + str((1/freq) * 1e+3) + ',' + 'fraction: ' + str(alpha) + ',' + 'prob: ' + str(probability) + ',' + 'templates: ' + str(total_templates_targed_search) + ',' + 'integration error percentage: ' + str((estimated_volume_integral_error/volume_integral_result) * 100) + ',' + 'coverage: ' + str(coverage) + ',' + 'mismatch: ' + str(mismatch) + ',' + 'phase: ' + str(max_initial_orbital_phase) + ',' + 'mass companion: ' + str(max_companion_mass) + '\n')

with open(output_path + output_filename + '_circular_orbit_gpu_format.txt', 'a') as outfile:
    for i in range(len(accepted_templates)):
        outfile.write(str(accepted_templates[i][0]) + ' ' + str(accepted_templates[i][1]) + ' ' + str(accepted_templates[i][2]) + ' ' + '\n')

    outfile.write(str(accepted_templates[0][0]) + ' ' + str(0.0) + ' ' + str(accepted_templates[0][2]) + ' ' + '\n')

