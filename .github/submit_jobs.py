import os
import time

modifiers = [""]  # , "_lim", "_lcdm"]

# runs = ["wl", "sn", "sl", "bao", "bao_cmb", "cmb", "sl_bao", "all"]

# n_cores = [8, 2, 4, 2, 8, 8, 2, 8]

run_specs = [
    # ["all", 8],
    # ["wl", 8],
    # ["bao_cmb", 8],
    ["cmb", 8],
    # ["sl", 4],
    # ["sn_bao", 2],
    # ["sn", 2],
    # ["bao", 2],
]

for r, n in run_specs:
    for m in modifiers:
        command = f"SBATCH --export=NUM_PROC={n},JOB_NAME={r}{m} submit.sh"
        print(command)
        os.system(command)
        time.sleep(1)
