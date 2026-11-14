"""Run the full replication of the network paper in order. Usage (from replication/code):  python run_all.py"""
import subprocess
import sys
import time

STEPS = ["01_clean_data.py", "00_geocode_check.py", "verify_original.py", "02_network.py", "03_rainfall.py",
         "04_indices_cointegration.py", "04b_johansen_cases.py", "05_tvecm.py", "06_rainfall_moderation.py", "07_fiedler_rolling.py",
         "08_rain_price_timing.py", "09_policy_bounds.py",
         ["02_network.py", "--shrinkage", "0.15", "--tag", "alpha015"],
         ["02_network.py", "--shrinkage", "0.50", "--tag", "alpha050"],
         ["02_network.py", "--pairwise", "--tag", "pairwise"],
         "02b_missing_data_check.py",
         "10_asymmetry_simam_partitions.py", "13_economic_interpretation.py", "14_community_iv.py", "15_monte_carlo.py",
         "11_figures.py", "12_paper_numbers.py"]

for step in STEPS:
    cmd = [sys.executable] + (step if isinstance(step, list) else [step])
    t0 = time.time(); print(">>", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True)
    print(f"   done in {time.time()-t0:.0f}s", flush=True)
print("All steps complete. Build the manuscript with:  cd ../paper && node build_paper.js")
