import argparse
import numpy as np
from multiprocessing import Pool
from functools import partial
from template_utils import compare_templates_with_fsearch, setup_logger

def check_template_vs_bank(suggested_idx, suggested_templates, bank_templates, freq, tobs, T2, T_inv, threshold, nmc,
                            f_search_range, coarse_threshold, coarse_step, fine_step, logger):
    candidate = suggested_templates[suggested_idx]
    for j, bank_tpl in enumerate(bank_templates):
        detstat, mismatch = compare_templates_with_fsearch(
            candidate, bank_tpl, freq, tobs, T2, T_inv, nmc,
            f_search_range, coarse_threshold, coarse_step, fine_step,
            logger=logger
        )
        logger.debug(f"Suggested template {suggested_idx} vs Bank template {j} → detstat = {detstat:.6f}, mismatch = {mismatch:.6f}")
        if detstat >= threshold:
            logger.debug(f"Template {suggested_idx} rejected due to overlap with bank template {j}")
            return False
    return True

def pair_check(bank_file, suggested_file, output_file, tobs, freq, mismatch_orbital, mismatch_spin_freq,
               nmc, log_interval, verbose, nproc,
               f_search_range, coarse_threshold, coarse_step, fine_step, merge_with_bank, keep_header=True):

    logger = setup_logger(verbose, log_interval)
    total_mismatch = mismatch_orbital + mismatch_spin_freq

    #Precompute constants that are used in the detection statistic
    T2 = tobs ** 2
    T_inv = 1.0 / tobs
    #Templates overlapping by more than 1 - mismatch are rejected
    threshold = (1.0 - total_mismatch) * T2

    logger.info(f"Loading bank templates from {bank_file}")
    with open(bank_file, 'r') as f:
        bank_lines = f.readlines()
    bank_header = [line for line in bank_lines if line.startswith("#")]
    bank_data = [line for line in bank_lines if not line.startswith("#") and line.strip()]
    bank_templates = np.loadtxt(bank_data, ndmin=2)

    logger.info(f"Loading suggested templates from {suggested_file}")
    with open(suggested_file, 'r') as f:
        suggested_lines = f.readlines()
    suggested_data = [line for line in suggested_lines if not line.startswith("#") and line.strip()]
    suggested_templates = np.loadtxt(suggested_data, ndmin=2)

    logger.info(f"Comparing {len(suggested_templates)} suggested templates against {len(bank_templates)} bank templates")

    with Pool(processes=nproc) as pool:
        check_func = partial(check_template_vs_bank,
                             suggested_templates=suggested_templates,
                             bank_templates=bank_templates,
                             freq=freq,
                             tobs=tobs,
                             T2=T2,
                             T_inv=T_inv,
                             threshold=threshold,
                             nmc=nmc,
                             f_search_range=f_search_range,
                             coarse_threshold=coarse_threshold,
                             coarse_step=coarse_step,
                             fine_step=fine_step,
                             logger=logger)
        results = pool.map(check_func, range(len(suggested_templates)))

    accepted_templates = suggested_templates[results]

    if merge_with_bank:
        final_templates = np.vstack([bank_templates, accepted_templates])
        logger.info(f"Merged {len(accepted_templates)} accepted templates with original bank. Final total: {len(final_templates)}. Rejected: {len(suggested_templates) - len(accepted_templates)}. Output saved to {output_file}")
    else:
        final_templates = accepted_templates
        logger.info(f"Writing only accepted templates. Count: {len(final_templates)}. Rejected: {len(suggested_templates) - len(final_templates)}. Output saved to {output_file}")

    with open(output_file, 'w') as fout:
        if keep_header and merge_with_bank:
            fout.writelines(bank_header)
        np.savetxt(fout, final_templates, fmt="%0.19f")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare suggested template bank against an existing one and keep only non-overlapping ones.")
    parser.add_argument("--bank_file", required=True, help="Template bank (existing templates)")
    parser.add_argument("--suggested_file", required=True, help="Suggested templates to test")
    parser.add_argument("--output_file", required=True, help="File to write accepted templates")
    parser.add_argument("--tobs", type=float, required=True, help="Observation time in seconds")
    parser.add_argument("--freq", type=float, required=True, help="Max Spin Frequency of template bank in Hz")
    parser.add_argument("--mismatch_orbital", type=float, default=0.1, help="Allowed mismatch due to orbital params")
    parser.add_argument("--mismatch_spin_freq", type=float, default=0.1, help="Allowed mismatch due to spin frequency scalloping")
    parser.add_argument("--nmc", type=int, default=10000, help="Number of Monte Carlo points for integration")
    parser.add_argument("--log_interval", type=int, default=60, help="Interval (s) between status logs. (default: 60s)")
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable detailed logging")
    parser.add_argument("--nproc", type=int, default=1, help="Number of processes to use for parallel processing.")
    parser.add_argument("--f_search_range", type=float, default=1.3, help="Frequency search range in Hz. Default is 1.3 Hz.")
    parser.add_argument("--coarse_threshold", type=float, default=0.2, help="Threshold to refine frequency resolution. Default is 0.2.")
    parser.add_argument("--coarse_step", type=float, default=1.0, help="Coarse step size for frequency search. Default is 1.0.")
    parser.add_argument("--fine_step", type=float, default=0.333, help="Fine step size for frequency search. Default is 0.333.")
    parser.add_argument("--no_merge_with_bank", action="store_false", dest="merge_with_bank", help="If set, DO NOT append accepted templates to input bank.")
    parser.add_argument("--no_header", action="store_true", help="If set, don't include the original header in output file")
    args = parser.parse_args()

    pair_check(
        bank_file=args.bank_file,
        suggested_file=args.suggested_file,
        output_file=args.output_file,
        tobs=args.tobs,
        freq=args.freq,
        mismatch_orbital=args.mismatch_orbital,
        mismatch_spin_freq=args.mismatch_spin_freq,
        nmc=args.nmc,
        log_interval=args.log_interval,
        verbose=args.verbose,
        nproc=args.nproc,
        f_search_range=args.f_search_range,
        coarse_threshold=args.coarse_threshold,
        coarse_step=args.coarse_step,
        fine_step=args.fine_step,
        merge_with_bank=args.merge_with_bank,
        keep_header=not args.no_header
    )