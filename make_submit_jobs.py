import sys
import pandas as pd
import subprocess
import numpy as np
import time

for monkey in ['V', 'W']:
	for session in range(4):
		for block in range(1,25):
			for seed in range(3):
				for pert_trial in range(1, 81):
					fit_string = f"python model_1p7.py {monkey} {session} {block} {seed} {pert_trial}"
					file_string = f'{monkey}_{session}_{block}_{seed}_{pert_trial}.sh'
					with open (file_string, 'w') as rsh:
						rsh.write('''#!/bin/bash''')
						rsh.write("\n")
						rsh.write('''#SBATCH --mem=32G''')
						rsh.write("\n")
						rsh.write('''#SBATCH --nodes=1''')
						rsh.write("\n")
						rsh.write('''#SBATCH --ntasks-per-node=1''')
						rsh.write("\n")
						rsh.write('''#SBATCH --time=0:30:0''')
						rsh.write("\n")
						rsh.write(fit_string)

for monkey in ['V', 'W']:
	for session in range(4):
		for block in range(1,25):
			for seed in range(3):
				for pert_trial in range(1, 81):
					submit_string = ["sbatch", f"{monkey}_{session}_{block}_{seed}_{pert_trial}.sh"]
					a = subprocess.run(submit_string)
					time.sleep(0.1)  # wait a second before next submission to help out SLURM system