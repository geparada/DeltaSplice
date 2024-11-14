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
    save_file = open(args.save_path, "w")

    if args.use_reference:
        with open(default_anno_file, "r") as f:
            default_anno_info = json.load(f)
            SortedKeys = {sp: {chr: {"+" : sorted([int(_) for _ in default_anno_info[sp][chr]["+"].keys()]),
                                     "-" : sorted([int(_) for _ in default_anno_info[sp][chr]["-"].keys()])}
                         for chr in default_anno_info[sp]}
                         for sp in default_anno_info}

    # Header output
    if "exon_start" not in input_file.keys():
        header = ("chrom,mut_position,ref,alt,strand,position,reference_acceptor_ssu,reference_donor_ssu,"
                  "pred_ref_acceptor_ssu,pred_ref_donor_ssu,pred_acceptor_deltassu,pred_donor_deltassu\n")
        if args.simple_output:
            header = "chrom,mut_position,strand,pred_acceptor_deltassu,pred_donor_deltassu\n"
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
                seq = seq + "N" * abs(seq_start)

            # Ensure the `ref` sequence matches
            ref_segment = seq[mut_pos - seq_start: mut_pos - seq_start + len(ref)]
            assert ref_segment.upper() == ref.upper(), (
                f"Expected ref segment {ref}, but found {ref_segment} at position {mut_pos}"
            )

            # Create mutated sequence with variable ref/alt lengths
            mutseq = seq[:mut_pos - seq_start] + alt.upper() + seq[mut_pos - seq_start + len(ref):]

            refmat = np.zeros((CL + EL, 3))
            if args.use_reference:
                species = args.genome.split("/")[-1].replace(".fa", "")
                assert species in SortedKeys, f"{args.genome} not in default reference"
                posidx = bisect_left(SortedKeys[species][chrom][strand], pos)
                startidx = bisect_left(SortedKeys[species][chrom][strand], seq_start)
                endidx = bisect_left(SortedKeys[species][chrom][strand], seq_end)
                for v in SortedKeys[species][chrom][strand][startidx:endidx]:
                    refmat[v - seq_start] = default_anno_info[species][chrom][strand][str(v)]
                refmat[np.isnan(refmat)] = 1e-3

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

            # Adjust prediction window for output
            write_window_start = pred_ref.shape[1] // 2 - args.window_size // 2
            write_window_end = pred_ref.shape[1] // 2 + args.window_size // 2

            if not args.simple_output:
                save_file.write(
                    f"{chrom},{mut_pos},{ref},{alt},{strand},"
                    + ";".join(map(str, [pos - args.window_size // 2 + i - write_window_start for i in range(write_window_start, write_window_end + 1)]))
                    + "," + ";".join(map(str, [pred_ref[0, i, 1] for i in range(write_window_start, write_window_end + 1)]))
                    + "," + ";".join(map(str, [pred_delta[0, i, 1] for i in range(write_window_start, write_window_end + 1)])) + "\n"
                )
            else:
                save_file.write(f"{chrom},{mut_pos},{ref},{alt},{strand},{pred_delta[0, write_window_start, 1]},{pred_delta[0, write_window_start, 2]}\n")
        
        save_file.close()

if __name__ == "__main__":
    main()
