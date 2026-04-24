import wandb

api = wandb.Api()
try:
    runs = api.runs("gunays-independent/lc0jaxhuman-jepa")
    print("Recent W&B Sweep Runs:")
    print("-" * 80)
    found = False
    for run in runs:
        if "muon-k4" in run.name:
            found = True
            print(f"Run Name: {run.name}")
            print(f"  State: {run.state}")
            print(f"  Created: {run.created_at}")
            try:
                summary = run.summary
                print(f"  Latest Step: {summary.get('_step', 'N/A')}")
                print(f"  Loss: {summary.get('loss', 'N/A')}")
                print(f"  Legality Loss: {summary.get('legality_loss', 'N/A')}")
            except Exception as e:
                print(f"  Could not fetch metrics: {e}")
            print("-" * 80)
    if not found:
        print("No muon-k4 runs found yet. They might still be starting.")
except Exception as e:
    print(f"Error: {e}")
