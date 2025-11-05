import sys
import pandas as pd
import subprocess
import numpy as np
import time

for monkey in ['V', 'W']:
	for session in range(3):
		for block in range(1,25):
			fit_string = f"python model_1p3.py {monkey} {session} {block}"
			file_string = f'job_nef_{monkey}_{session}_{block}.sh'
			with open (file_string, 'w') as rsh:
				rsh.write('''#!/bin/bash''')
				rsh.write("\n")
				rsh.write('''#SBATCH --mem=8G''')
				rsh.write("\n")
				rsh.write('''#SBATCH --nodes=1''')
				rsh.write("\n")
				rsh.write('''#SBATCH --ntasks-per-node=1''')
				rsh.write("\n")
				rsh.write('''#SBATCH --time=0:10:0''')
				rsh.write("\n")
				rsh.write(fit_string)

for monkey in ['V', 'W']:
	for session in range(3):
		for block in range(1,25):
			submit_string = ["sbatch", f"job_nef_{monkey}_{session}_{block}.sh"]
			a = subprocess.run(submit_string)
			time.sleep(1)  # wait a few seconds before next submission to help out SLURM system