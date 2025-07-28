import json
import math
import re
import sys


for file in sys.argv[1:]:
    bp_iters = re.match(r".*results\.bp=(\d+)\.json", file).group(1) # type:ignore
    with open(file) as f:
        results = json.load(f)
        for result in results:
            cenested = -result["cross_entropy(interleaved, nested)"]
            cerandom = -result["cross_entropy(interleaved, random_nested) mean"]
            cestdev = math.sqrt(result["cross_entropy(interleaved, random_nested) variance"] / result["cross_entropy(interleaved, random_nested) count"])
            print("{} & {} & {} & {:.4f} $\pm$ {:.4f} \\\\".format(
                result["config"]["seed"],
                bp_iters,
                f"{cenested:.4f}" if cenested >= cerandom - 3 * cestdev else f"\\textbf{{{cenested:.4f}}}",
                cerandom,
                cestdev))
