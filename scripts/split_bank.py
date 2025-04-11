import os
import argparse

def split_template_bank(input_file, n_splits, output_dir, keep_header=True, dry_run=False):
    os.makedirs(output_dir, exist_ok=True)

    # Separate header and data
    with open(input_file, 'r') as f:
        lines = f.readlines()
    header_lines = [line for line in lines if line.startswith("#")]
    data_lines = [line for line in lines if not line.startswith("#") and line.strip()]

    total_templates = len(data_lines)
    chunk_size = total_templates // n_splits
    extras = total_templates % n_splits

    print(f"Splitting {total_templates} templates into {n_splits} parts...")
    print(f"Each file will contain ~{chunk_size} templates (+1 for the first {extras} files)")

    index = 0
    for i in range(n_splits):
        count = chunk_size + (1 if i < extras else 0)
        out_path = os.path.join(output_dir, f"bank_split_{i:04d}.txt")
        if dry_run:
            print(f"[DRY RUN] Would write {count} templates to {out_path}")
        else:
            with open(out_path, 'w') as fout:
                if keep_header:
                    fout.writelines(header_lines)
                fout.writelines(data_lines[index:index + count])
        index += count

    if not dry_run:
        print(f"\n✅ Done. Split into {n_splits} files under {output_dir}")
    else:
        print(f"\n✅ Dry run complete. No files were written.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a template bank into multiple parts for parallel filtering.")
    parser.add_argument("--input_file", required=True, help="Full template bank to be split")
    parser.add_argument("--n_splits", type=int, required=True, help="How many subsets to split into")
    parser.add_argument("--output_dir", required=True, help="Output directory to store split files")
    parser.add_argument("--no_header", action="store_true", help="If set, don't include the original header in each split")
    parser.add_argument("--dry_run", action="store_true", help="If set, only print what would be done")
    args = parser.parse_args()

    split_template_bank(
        input_file=args.input_file,
        n_splits=args.n_splits,
        output_dir=args.output_dir,
        keep_header=not args.no_header,
        dry_run=args.dry_run
    )

