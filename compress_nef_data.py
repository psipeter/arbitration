import pandas as pd
import re
from pathlib import Path

def process_nef_data(folder_path="data/nef/", do_full=False):
    path = Path(folder_path)
    
    # Define output paths
    output_full = path / "nef_data_full.pkl.xz"
    output_num  = path / "nef_data.pkl.xz"

    # --- PART 1: Process "_full.pkl" files ---
    # Captures files like: trial_1_full.pkl, trial_2_full.pkl
    if do_full:
        full_files = list(path.glob("*_full.pkl"))
        
        if full_files:
            print(f"Concatenating {len(full_files)} '_full.pkl' files...")
            df_full = pd.concat((pd.read_pickle(f) for f in full_files), ignore_index=True)
            
            print(f"Compressing (xz) and saving to {output_full}...")
            df_full.to_pickle(output_full, compression="xz")
            
            # Explicitly clear memory
            del df_full 
        else:
            print("No '_full.pkl' files found.")

    # --- PART 2: Process "(number).pkl" files ---
    # Captures files like: trial_1.pkl, 10.pkl
    # Regex ensures the filename ends in a digit before the extension
    all_pkls = list(path.glob("*.pkl"))
    num_files = [f for f in all_pkls if re.search(r'\d\.pkl$', str(f))]

    if num_files:
        print(f"\nConcatenating {len(num_files)} numbered '.pkl' files...")
        df_num = pd.concat((pd.read_pickle(f) for f in num_files), ignore_index=True)
        
        print(f"Compressing (xz) and saving to {output_num}...")
        df_num.to_pickle(output_num, compression="xz")
        
        del df_num
    else:
        print("No numbered '.pkl' files found.")

    print("\nProcess complete.")

if __name__ == "__main__":
    process_nef_data()