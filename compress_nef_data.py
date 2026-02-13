import pandas as pd
import re
import h5py
from pathlib import Path

def process_nef_data(folder_path="data/nef"):
    path = Path(folder_path)
    output_probes = path / "nef_data_probes.pkl.xz"
    output_values  = path / "nef_data_values.pkl.xz"

    probe_files = list(path.glob("*_probes.pkl"))
    if probe_files:
        print(f"Concatenating {len(probe_files)} '_probes.pkl' files...")
        df_probe = pd.concat((pd.read_pickle(f) for f in probe_files), ignore_index=True)
        print(f"Compressing (xz) and saving to {output_probes}...")
        df_probe.to_pickle(output_probes, compression="xz")
        del df_probe 
    else:
        print("No '_probes.pkl' files found.")

    value_files = list(path.glob("*_values.pkl"))
    if value_files:
        print(f"Concatenating {len(value_files)} '_values.pkl' files...")
        df_value = pd.concat((pd.read_pickle(f) for f in value_files), ignore_index=True)
        print(f"Compressing (xz) and saving to {output_values}...")
        df_value.to_pickle(output_values, compression="xz")
        del df_value 
    else:
        print("No '_values.pkl' files found.")

    print("\nProcess complete.")

if __name__ == "__main__":
    process_nef_data()