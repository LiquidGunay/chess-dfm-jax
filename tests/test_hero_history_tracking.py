import json

from research.hero_history_tracking import JsonlTail, scalar_event


def test_tail_does_not_lose_incomplete_lines(tmp_path):
    path = tmp_path/"metrics.jsonl"
    tail = JsonlTail(path)
    assert tail.read() == []
    path.write_text('{"update":1}\n{"up')
    assert tail.read() == [{"update":1}]
    assert tail.read() == []
    with path.open("a") as stream:stream.write('date":2}\n')
    assert tail.read() == [{"update":2}]


def test_scalar_events_have_comparable_axes_and_no_metadata():
    event = scalar_event({"update":100,"examples":102400,"loss":1.5,"config":{"token":"never"}},"train")
    assert event["train/loss"] == 1.5
    assert "train/config" not in event
    assert scalar_event({"update":100,"examples":102400,"metrics":{"loss":2}},"val")["val/loss"] == 2
