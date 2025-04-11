import argparse
import numpy as np
import logging
import time
from template_utils import compare_templates_with_fsearch, setup_logger

def self_check(input_file, output_file, tobs, freq, mismatch_orbital, mismatch_spin_freq, nmc,
               log_interval, verbose, f_search_range, coarse_threshold, coarse_step, fine_step, keep_header=True):
    logger = setup_logger(verbose, log_interval)

    with open(input_file, 'r') as f:
        lines = f.readlines()
    header_lines = [line for line in lines if line.startswith("#")]
    data_lines = [line for line in lines if not line.startswith("#") and line.strip()]

    templates = np.loadtxt(data_lines, ndmin=2)
    N = len(templates)
    total_mismatch = mismatch_orbital + mismatch_spin_freq
    # Precompute constants that are used in the detection statistic
    T2 = tobs ** 2
    T_inv = 1.0 / tobs
    # Templates overlapping by more than 1 - mismatch are rejected
    threshold = (1.0 - total_mismatch) * T2

    keep_flags = np.ones(N, dtype=bool)
    logger.info(f"Running lower-triangular self-check on {N} templates")
    logger.debug(f"Computed constants → T2: {T2}, T_inv: {T_inv:.6e}, Threshold: {threshold:.6f}")

    for i in range(N):
        if not keep_flags[i]:
            continue
        for j in range(i + 1, N):
            if not keep_flags[j]:
                continue
            detstat, mismatch = compare_templates_with_fsearch(
                templates[i], templates[j], freq, tobs, T2, T_inv, nmc,
                f_search_range, coarse_threshold, coarse_step, fine_step,
                logger=logger
            )
            logger.debug(f"Compared template {i} vs {j} → detstat = {detstat:.6f}, mismatch = {mismatch:.6f}")
            if detstat >= threshold:
                logger.debug(f"Template {j} rejected due to overlap with template {i}")
                keep_flags[j] = False
        if i % 10 == 0:
            logger.info(f"Checked {i+1}/{N} templates — {np.sum(keep_flags)} retained")

    with open(output_file, 'w') as fout:
        if keep_header:
            fout.writelines(header_lines)
        np.savetxt(fout, templates[keep_flags], fmt="%0.19f")

    logger.info(f"Self-check done. Retained {np.sum(keep_flags)} templates out of {N}. Rejected {N - np.sum(keep_flags)} templates. Output saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform self-filtering via pairwise comparisons on one template bank.")
    parser.add_argument("--input_file", required=True, help="Template bank (existing templates)")
    parser.add_argument("--output_file", required=True, help="File to write accepted templates")
    parser.add_argument("--tobs", type=float, required=True, help="Observation time in seconds")
    parser.add_argument("--freq", type=float, required=True, help="Max Spin Frequency of template bank in Hz")
    parser.add_argument("--mismatch_orbital", type=float, default=0.1, help="Allowed mismatch due to orbital params")
    parser.add_argument("--mismatch_spin_freq", type=float, default=0.1, help="Allowed mismatch due to spin frequency scalloping")
    parser.add_argument("--nmc", type=int, default=10000, help="Number of Monte Carlo points for integration")
    parser.add_argument("--log_interval", type=int, default=60, help="Interval (s) between status logs. (default: 60s)")
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable detailed logging")
    parser.add_argument("--f_search_range", type=float, default=1.3, help="Frequency search range in Hz. Default: 1.3Hz")
    parser.add_argument("--coarse_threshold", type=float, default=0.2, help="Threshold to refine frequency resolution. Default: 0.2")
    parser.add_argument("--coarse_step", type=float, default=1.0, help="Coarse step size for frequency search. Default: 1.0")
    parser.add_argument("--fine_step", type=float, default=0.333, help="Fine step size for frequency search. Default: 0.333")
    parser.add_argument("--no_header", action="store_true", help="If set, don't include the original header in output file")
    args = parser.parse_args()

    self_check(
        input_file=args.input_file,
        output_file=args.output_file,
        tobs=args.tobs,
        freq=args.freq,
        mismatch_orbital=args.mismatch_orbital,
        mismatch_spin_freq=args.mismatch_spin_freq,
        nmc=args.nmc,
        log_interval=args.log_interval,
        verbose=args.verbose,
        f_search_range=args.f_search_range,
        coarse_threshold=args.coarse_threshold,
        coarse_step=args.coarse_step,
        fine_step=args.fine_step,
        keep_header=not args.no_header
    )
