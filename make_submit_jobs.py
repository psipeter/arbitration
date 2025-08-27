import sys
import pandas as pd
import subprocess
import numpy as np
import time

sessions = range(3)

for monkey in ['V', 'W']:
	for session in sessions:
		fit_string = f"python model_1p2.py {monkey} {session} load fitting"
		file_string = f'job_{monkey}{session}.sh'
		with open (file_string, 'w') as rsh:
			rsh.write('''#!/bin/bash''')
			rsh.write("\n")
			rsh.write('''#SBATCH --mem=8G''')
			rsh.write("\n")
			rsh.write('''#SBATCH --nodes=1''')
			rsh.write("\n")
			rsh.write('''#SBATCH --ntasks-per-node=1''')
			rsh.write("\n")
			rsh.write('''#SBATCH --time=12:0:0''')
			rsh.write("\n")
			rsh.write(fit_string)

	for session in sessions:
		submit_string = ["sbatch", f"job_{monkey}{session}.sh"]
		a = subprocess.run(submit_string)
		time.sleep(1)  # wait a few seconds before next submission to help out SLURM system