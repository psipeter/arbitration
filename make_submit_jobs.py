import sys
import pandas as pd
import subprocess
import numpy as np
import time

monkeys = ['V', 'W']
sessions = range(4)
blocks = range(1,25)
seeds = range(3)
for monkey in monkeys:
	for session in sessions:
		for block in blocks:
			for seed in seeds:
				fit_string = f"python model_1p7.py {monkey} {session} {block} {seed}"
				file_string = f'{monkey}_{session}_{block}_{seed}.sh'
				with open (file_string, 'w') as rsh:
					rsh.write('''#!/bin/bash''')
					rsh.write("\n")
					rsh.write('''#SBATCH --mem=32G''')
					rsh.write("\n")
					rsh.write('''#SBATCH --nodes=1''')
					rsh.write("\n")
					rsh.write('''#SBATCH --ntasks-per-node=1''')
					rsh.write("\n")
					rsh.write('''#SBATCH --time=1:00:0''')
					rsh.write("\n")
					rsh.write(fit_string)

for monkey in monkeys:
	for session in sessions:
		for block in blocks:
			for seed in seeds:
				submit_string = ["sbatch", f"{monkey}_{session}_{block}_{seed}.sh"]
				a = subprocess.run(submit_string)
				time.sleep(0.2)  # wait a second before next submission to help out SLURM system