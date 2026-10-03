"""Loss tracking separate from the optimizer; JSONL remains authoritative."""
import json
import math
import os
from pathlib import Path


def scalar_event(record, split):
    metrics = record["metrics"] if split == "val" else record
    if split == "val" and record.get("evaluation_examples") == 8128:
        split = "val_full"
    event = {"examples": int(record["examples"]), "update": int(record["update"])}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            event[f"{split}/{key}"] = value
    return event


class JsonlTail:
    def __init__(self, path):
        self.path, self.offset = Path(path), 0

    def read(self):
        if not self.path.exists():
            return []
        result = []
        with self.path.open() as stream:
            stream.seek(self.offset)
            while True:
                line = stream.readline()
                if not line or not line.endswith("\n"):
                    break
                result.append(json.loads(line))
                self.offset = stream.tell()
        return result


class LossTracker:
    def __init__(self, directory, *, run_id, name, config):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.events = self.directory/f"{run_id}.events.jsonl"
        self.run, self.error = None, None
        self.finished = False
        mode = "online" if os.environ.get("WANDB_API_KEY") else "offline"
        try:
            import wandb
            self.run = wandb.init(project=os.environ.get("WANDB_PROJECT", "chess-dfm-history"),
                entity=os.environ.get("WANDB_ENTITY") or None, group="HH-002-history-comparison",
                id=run_id, name=name, config=config, dir=str(self.directory), mode=mode,
                save_code=False, settings=wandb.Settings(disable_git=True, init_timeout=30))
            self.run.define_metric("examples")
            self.run.define_metric("train/*", step_metric="examples")
            self.run.define_metric("val/*", step_metric="examples")
            self.run.define_metric("val_full/*", step_metric="examples")
        except Exception as exc:
            self.error = type(exc).__name__
        self.receipt = {"id":run_id, "name":name, "requested_mode":mode,
                        "wandb_started": self.run is not None, "error_type":self.error,
                        "url":self.run.url if self.run and mode == "online" else None}
        (self.directory/f"{run_id}.tracking.json").write_text(json.dumps(self.receipt,indent=2)+"\n")
        print(json.dumps({"loss_tracking":self.receipt}),flush=True)

    def log(self, record, split):
        event = scalar_event(record, split)
        with self.events.open("a") as stream:
            stream.write(json.dumps(event,sort_keys=True)+"\n")
        if self.run:
            try:
                self.run.log(event)
            except Exception as exc:
                self.error = type(exc).__name__
                self.receipt["logging_error_type"] = self.error

    def finish(self):
        if self.finished:
            return
        self.finished = True
        if self.run:
            try:
                self.run.finish()
            except Exception as exc:
                self.receipt["finish_error_type"] = type(exc).__name__
        (self.directory/f"{self.receipt['id']}.tracking.json").write_text(json.dumps(self.receipt,indent=2)+"\n")


def import_historical_curves(directory, baseline, validation_path, *, run_id, config):
    tracker = LossTracker(directory, run_id=run_id, name="Hero1 — no history (historical training)", config=config)
    events = []
    for split, path in (("train", Path(baseline)/"metrics.jsonl"), ("val", Path(validation_path))):
        for record in JsonlTail(path).read():
            events.append((record["update"], split, record))
    try:
        for _, split, record in sorted(events, key=lambda item:(item[0],item[1])):
            tracker.log(record,split)
    finally:
        tracker.finish()
    return tracker.receipt
