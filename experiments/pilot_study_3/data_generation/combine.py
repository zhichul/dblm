import argparse
import math
import sys

import torch
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--infiles", nargs="+", required=True)
parser.add_argument("--outfile", required=True)
parser.add_argument("--first", type=int, default=math.inf, required=False)
args = parser.parse_args()
infiles = args.infiles
outfile = args.outfile
if not input(f"Combining into {outfile} using {infiles}. Hit enter to continue.") == "":
    exit(0)
d = dict()
reached = False
for file in tqdm.tqdm(infiles, leave=False):
    dd = torch.load(file)
    for key in dd:
        comb = dd[key] if key not in d else torch.cat([d[key], dd[key]], dim=0)
        if comb.size(0) > args.first:
            comb = comb[:args.first]
            reached = True
        d[key] = comb
    if reached:
        break
torch.save(d, outfile)
