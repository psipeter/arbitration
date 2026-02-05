import sys
import pandas as pd
import subprocess
import numpy as np
import time

monkeys = ['V', 'W']
sessions = range(4)
blocks = range(1,25)
seeds = range(1)
pert_times = np.arange(1, 91, 10)
for monkey in monkeys:
	for session in sessions:
		for block in blocks:
			for seed in seeds:
				for p in range(len(pert_times)-1):
					pert_start = pert_times[p]
					pert_end = pert_times[p+1]
					fit_string = f"python model_1p7.py {monkey} {session} {block} {seed} {pert_start} {pert_end}"
					file_string = f'{monkey}_{session}_{block}_{seed}_{pert_start}.sh'
					with open (file_string, 'w') as rsh:
						rsh.write('''#!/bin/bash''')
						rsh.write("\n")
						rsh.write('''#SBATCH --mem=16G''')
						rsh.write("\n")
						rsh.write('''#SBATCH --nodes=1''')
						rsh.write("\n")
						rsh.write('''#SBATCH --ntasks-per-node=1''')
						rsh.write("\n")
						rsh.write('''#SBATCH --time=0:30:0''')
						rsh.write("\n")
						rsh.write(fit_string)

for monkey in monkeys:
	for session in sessions:
		for block in blocks:
			for seed in seeds:
				for p in range(len(pert_times)-1):
					pert_start = pert_times[p]
					submit_string = ["sbatch", f"{monkey}_{session}_{block}_{seed}_{pert_start}.sh"]
					a = subprocess.run(submit_string)
					time.sleep(0.1)  # wait a second before next submission to help out SLURM system