# Template Bank Generator for Binary Pulsar Searches

This repository contains scripts to generate random and stochastic template banks for binary pulsar searches in radio observations. These template banks are compatible with [peasoup (Keplerian branch)](https://github.com/vishnubk/peasoup/tree/keplerian) for GPU-based searches.

## Installation

To avoid dependency issues, use the provided Apptainer/Singularity container:

```bash
apptainer pull docker://vishnubk/template_bank_generator
```

---

## Overview of Supported Searches

### 1. Circular Orbit Searches (3-parameter search)

* **Pb**: Orbital period
* **τ = asini / c**: Projected semi-major axis in light seconds
* **φ**: Initial orbital phase

In the code:

* `omega = 2π / Pb`
* Parameters: `omega`, `tau`, `phi`

### 2. Elliptical Orbit Searches (5-parameter search)

* All parameters from circular orbit
* **e**: Eccentricity
* **ω̄**: Longitude of periastron

---

## Step 1: Estimate Number of Required Templates

Use the script:

```bash
python scripts/calculate_required_random_templates_circular_orbit_searches.py
```

This uses the algorithm from [Messenger et al. 2008](https://ui.adsabs.harvard.edu/abs/2009PhRvD..79j4017M/abstract) and generates a random template bank via a Metropolis-Hastings algorithm.

### Arguments

```text
-h, --help                Show help message
-o, --output_path         Directory to save results
-t, --obs_time            Observation time (minutes)
-p, --porb_low            Minimum orbital period (minutes)
-P, --porb_high           Maximum orbital period (minutes)
-c, --max_comp_mass       Maximum companion mass (M☉)
-d, --min_pulsar_mass     Minimum pulsar mass (M☉)
-s, --spin_period         Fastest pulsar spin period (ms)
-f, --fraction            Probability fraction for inclination angles
-b, --coverage            Desired template-bank coverage
-m, --mismatch            Maximum mismatch allowed
-n, --ncpus               Number of CPUs to use
-i, --nmc                 Monte Carlo iterations
-file, --output_filename  Output filename (optional)
```

### Example

```bash
python calculate_required_random_templates_circular_orbit_searches.py \
    -t 4.5 -p 11 -P 45 -c 1.6 -d 1.2 -s 2.5 -f 0.5 -b 0.9 -m 0.2 -n 48 -i 100000
```

If `-file` is not provided, the script prints the required number of templates without generating a file—useful for feasibility scans.

#### Sample Output

```
=== Circular Orbit Search Summary ===
Volume Integral: 2101.10
Volume Integral Error: 24.03
Coverage: 0.900
Mismatch: 0.200
Observation Time: 4.50 min
Pulsar Mass: 1.20 M☉
Companion Mass: 1.60 M☉
Orbital Period: 11.00–45.00 min
Spin Period: 2.50 ms
Required templates per DM trial: 12911
```

Use this value to estimate total run time:

```text
Total trials = N_templates × N_DM × N_beams × N_pointings
```

Estimate wall time:

```text
Wall time (days) = (search_time_per_trial × total_trials) / (N_GPUs × 86400)
```

### Search time Per Trial Benchmark for [peasoup](https://github.com/vishnubk/peasoup/tree/keplerian) on A100 (80 GB GPU)

| FFT Size | 8 Harmonics | 16 Harmonics | 32 Harmonics |
| -------- | ----------- | ------------ | ------------ |
| 2²⁴      | 0.00087 s   | 0.00126 s    | 0.00139 s    |
| 2²⁵      | 0.00173 s   | 0.00252 s    | 0.00277 s    |
| 2²⁶      | 0.00327 s   | 0.00477 s    | 0.00525 s    |

---

## Step 2: Generate the Template Bank

Once you've finalized the parameter space and number of templates, you can generate the actual bank using either:

### Option 1: Metropolis-Hastings (for small banks)

Add `-file <filename>` to the previous script to generate the bank directly.

### Option 2: Emcee MCMC (recommended for >20,000 templates)

This uses [emcee](https://emcee.readthedocs.io/en/stable/) to parallelize sampling.

```bash
python scripts/construct_random_template_bank_emcee.py \
    -t 4.5 -p 11 -P 45 -c 1.6 -d 1.2 -s 2.5 -f 0.5 -b 0.9 -m 0.2 -n 48 \
    -file my_template_bank -z 12911
```

* `-file`: Output filename prefix
* `-z`: Number of templates (from step 1)

Outputs:

* `my_template_bank_peasoup_format.txt` — ready to use with `peasoup`
* Corner plot showing distribution of orbital templates

---

## Citation

If you use this software in your research, please cite:

> **Balakrishnan et al. (2022)**, [MNRAS, 511, 1265](https://ui.adsabs.harvard.edu/abs/2022MNRAS.511.1265B/abstract)

---

## Issues and Contributions

If you encounter any bugs, have questions, or want to contribute features:

* [Open an issue]([https://github.com/your-org/template_bank_generator/issues](https://github.com/erc-compact/template_bank_generator/issues))
* [Submit a pull request]([https://github.com/your-org/template_bank_generator/pulls](https://github.com/erc-compact/template_bank_generator/pulls))




