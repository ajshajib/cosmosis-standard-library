import os
import time

modifiers = [""]  # "_lim", "_lcdm"]  # ""

# runs = ["wl", "sn", "sl", "bao", "bao_cmb", "cmb", "sl_bao", "all"]

# n_cores = [8, 2, 4, 2, 8, 8, 2, 8]

run_specs = [
    # ["all", 8],
    ["wl", 8],
    # ["bao_cmb", 8],
    # ["cmb", 8],
    # ["sl", 4],
    # ["sn_bao", 2],
    # ["sn", 2],
    # ["bao", 2],
]

for mod in modifiers:
    for run, n in run_specs:
        n_core = n * 48
        command = f"sbatch --job-name={run}{mod} --ntasks={n_core} --export=NUM_PROC={n_core},MOD_NAME={mod},RUN_NAME={run} submit.sh"
        print(command)
        os.system(command)
        time.sleep(1)
