import os
import sys
import wandb
import datetime

def check_wandb_runs():
    api = wandb.Api()
    # Query runs from the chess_dfm_jax-jepa project
    try:
        runs = api.runs("chess_dfm_jax-jepa")
        print("Recent W&B Runs (Past 24 hours):")
        print("-" * 80)
        
        now = datetime.datetime.now(datetime.timezone.utc)
        
        found_runs = False
        for run in runs:
            # Check if run was updated in the last 24h
            created_at = datetime.datetime.strptime(run.created_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)
            if (now - created_at).total_seconds() > 24 * 3600 and run.state != "running":
                continue
                
            if "v21-dfm" in run.name:
                found_runs = True
                print(f"Run Name: {run.name}")
                print(f"  State: {run.state}")
                print(f"  Created: {run.created_at}")
                print(f"  URL: {run.url}")
                
                # Fetch recent metrics
                try:
                    summary = run.summary
                    step = summary.get("_step", "N/A")
                    loss = summary.get("loss", "N/A")
                    jepa_loss = summary.get("jepa_loss", "N/A")
                    valid_frac = summary.get("valid_fraction", "N/A")
                    print(f"  Latest Step: {step}")
                    print(f"  Loss: {loss}")
                    print(f"  Valid Fraction: {valid_frac}")
                except Exception as e:
                    print(f"  Could not fetch metrics: {e}")
                print("-" * 80)
                
        if not found_runs:
            print("No v21-dfm runs found in the recent W&B history.")
            
    except Exception as e:
        print(f"Error querying W&B: {e}")

if __name__ == "__main__":
    check_wandb_runs()
