import os
import argparse
import torch
import copy
import pandas as pd
from deltasplice.constant import default_model_paths, model, Fapath, EL, CL, SeqTable, repdict, IN_MAP, default_anno_file
from pyfasta import Fasta
import numpy as np
import json
from bisect import bisect_left

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to input file")
    parser.add_argument("--save_path", help="Path to output file")
    parser.add_argument("--use_reference", default=False, action="store_true")
    parser.add_argument("--simple_output", default=False, action="store_true")
    parser.add_argument("--window_size", type=int, default=200)
    parser.add_argument("--genome", help="Reference genome")
    args = parser.parse_args()

    Models = [copy.deepcopy(model) for _ in default_model_paths]
    [m.load_state_dict(torch.load(b)) for m, b in zip(Models, default_model_paths)]
    
    input_file = pd.read_csv(args.data_path)
    reference_genome = Fasta(os.path.join(Fapath, args.genome + ".fa"))
    
    # Load annotation info if using reference
    if args.use_reference:
        with open(default_anno_file, "r") as f:
            default_anno_info = json.load(f)
            SortedKeys = {sp: {chr: {"+" : sorted([int(_) for _ in default_anno_info[sp][chr]["+"].keys()]),
                                     "-" : sorted([int(_) for _ in default_anno_info[sp][chr]["-"].keys()])}
                         for chr in default_anno_info[sp]}
                         for sp in default_anno_info}

    with open(args.save_path, "w") as save_file:
        header = ("chrom,mut_position,ref,alt,strand,reference_acceptor_ssu,reference_donor_ssu,"
                  "pred_ref_acceptor_ssu,pred_ref_donor_ssu,pred_acceptor_deltassu,pred_donor_deltassu\n")
        save_file.write(header)

        for chrom, mut_pos, ref, alt, strand in zip(input_file["chrom"], input_file["mut_position"], input_file["ref"], input_file["alt"], input_file["strand"]):
            pos = mut_pos
            seq_start = pos - (EL + CL) // 2
            seq_end = seq_start + EL + CL

            # Extract sequence around the mutation
            seq = reference_genome[chrom][max(seq_start, 0):min(seq_end, len(reference_genome[chrom]))].upper()
            if seq_start < 0:
                seq = "N" * abs(seq_start) + seq
            if seq_end > len(reference_genome[chrom]):
                seq += "N" * abs(seq_start)

            # Ensure the `ref` sequence matches
            ref_segment = seq[mut_pos - seq_start: mut_pos - seq_start + len(ref)]
            if ref_segment.upper() != ref.upper():
                print(f"Warning: ref segment mismatch for {chrom} at position {mut_pos}")
                continue

            # Retrieve reference SSU values if using the reference annotation
            if args.use_reference:
                species = args.genome.split("/")[-1].replace(".fa", "")
                if species not in default_anno_info:
                    print(f"Warning: {args.genome} not found in default reference.")
                    continue
                
                acceptor_ssu = default_anno_info[species][chrom][strand].get(str(pos), [0, 0, 0])[1]
                donor_ssu = default_anno_info[species][chrom][strand].get(str(pos), [0, 0, 0])[2]
            else:
                # Default reference SSU values if not using the annotation
                acceptor_ssu = 0.0
                donor_ssu = 0.0

            # Create mutated sequence
            mutseq = seq[:mut_pos - seq_start] + alt.upper() + seq[mut_pos - seq_start + len(ref):]
            refmat = np.zeros((CL + EL, 3))
            refmat[mut_pos - seq_start] = [1 - acceptor_ssu - donor_ssu, acceptor_ssu, donor_ssu]

            if strand == "-":
                seq = [repdict[_] for _ in seq][::-1]
                mutseq = [repdict[_] for _ in mutseq][::-1]
                refmat = refmat[::-1]

            seq = IN_MAP[[SeqTable[_] for _ in seq]][:, :4]
            mutseq = IN_MAP[[SeqTable[_] for _ in mutseq]][:, :4]
            refmat[:, 0] = 1 - refmat[:, 1:].sum(-1)
            refmat = refmat[EL // 2: EL // 2 + CL].copy()

            d = {
                "X": torch.tensor(seq)[None],
                "mutX": torch.tensor(mutseq)[None],
                "single_pred_psi": torch.tensor(refmat)[None]
            }
            use_ref = not args.use_reference
            pred = [m.predict(d, use_ref=use_ref) for m in Models]
            pred_ref = sum([v["single_pred_psi"] for v in pred]) / len(pred)
            pred_delta = sum([v["mutY"] for v in pred]) / len(pred) - pred_ref

            # Summarize predictions for acceptor and donor signals
            pred_acceptor_ref = pred_ref[:, :, 1].mean().item()
            pred_donor_ref = pred_ref[:, :, 2].mean().item()
            pred_acceptor_delta = pred_delta[:, :, 1].mean().item()
            pred_donor_delta = pred_delta[:, :, 2].mean().item()

            # Write output in the expected format
            save_file.write(f"{chrom},{mut_pos},{ref},{alt},{strand},"
                            f"{acceptor_ssu},{donor_ssu},"
                            f"{pred_acceptor_ref},{pred_donor_ref},{pred_acceptor_delta},{pred_donor_delta}\n")

    print("Prediction completed and saved.")

if __name__ == "__main__":
    main()

