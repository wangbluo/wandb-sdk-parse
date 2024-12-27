import json
import os
import time
from glob import glob
from typing import Any, Dict, Tuple

from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal import datastore
import multiprocessing as mp
import os
import tempfile
import time

import wandb


def robust_scan(ds):
    try:
        return ds.scan_data()
    except AssertionError as e:
        if ds.in_last_block():
            return None
        else:
            raise e


def parse_pb(data):
    pb = wandb_internal_pb2.Record()
    pb.ParseFromString(data)
    record_type = pb.WhichOneof("record_type")
    return pb, record_type


def parse_pb_items(items) -> Dict[str, Any]:
    data = {}
    for item in items:
        key = item.key
        if len(key) == 0:
            if len(item.nested_key) > 1:
                raise ValueError("nested_key has more than one element")
            key = item.nested_key[0]
        data[key] = item.value_json
    return data


def parse_history(pb) -> Tuple[int, Dict[str, Any]]:
    step = pb.history.step.num
    data = parse_pb_items(pb.history.item)
    for k, v in data.items():
        if k in ("_timestamp", "_runtime"):
            data[k] = float(v)
        elif k == "_step":
            data[k] = int(v)
    return step, data


def parse_stats(pb) -> Tuple[int, Dict[str, Any]]:
    timestamp = pb.stats.timestamp.seconds
    data = parse_pb_items(pb.stats.item)
    return timestamp, data


def parse_summary(pb) -> Dict[str, Any]:
    data = parse_pb_items(pb.summary.update)
    for k, v in data.items():
        if k in ("_timestamp", "_runtime"):
            data[k] = float(v)
        elif k == "_step":
            data[k] = int(v)
    return data


def parse_wandb_file(path: str):
    ds = datastore.DataStore()
    ds.open_for_scan(path)
    history = {}
    stats = {}
    summary = None
    while True:
        data = robust_scan(ds)
        if data is None:
            break
        pb, record_type = parse_pb(data)
        if record_type == "history":
            step, data = parse_history(pb)
            history[step] = data
        elif record_type == "stats":
            timestamp, data = parse_stats(pb)
            stats[timestamp] = data
        elif record_type == "summary":
            # use the last valid summary
            try:
                summary = parse_summary(pb)
            except ValueError:
                pass
    return {"history": history, "stats": stats, "summary": summary}


def get_wandb_file_path(wandb_root_dir: str) -> str:
    if wandb_root_dir.endswith("/"):
        wandb_root_dir = wandb_root_dir[:-1]
    files = glob(f"{wandb_root_dir}/latest-run/*.wandb")
    assert len(files) == 1, f"Expect to find one wandb file in {wandb_root_dir}, but found {len(files)}"
    return files[0]


def get_wandb_metadata(wandb_root_dir: str) -> Dict[str, Any]:
    metadata_file_path = os.path.join(wandb_root_dir, "latest-run/files/wandb-metadata.json")
    with open(metadata_file_path) as f:
        metadata = json.load(f)
    return metadata


def get_wandb_run(wandb_root_dir: str) -> Dict[str, Any]:
    wandb_file_path = get_wandb_file_path(wandb_root_dir)
    return parse_wandb_file(wandb_file_path)


def get_wandb_run_until(wandb_root_dir: str, try_until: int = 5) -> Dict[str, Any]:
    failed_cnt = 0
    while True:
        try:
            return get_wandb_run(wandb_root_dir)
        except AssertionError as e:
            failed_cnt += 1
            if failed_cnt >= try_until:
                raise e
            time.sleep(0.5)
            


os.environ["WANDB_MODE"] = "offline"

TOTAL_STEPS = 30



def test_wandb():
    
    wandb_root_dir = "/home/wangbinluo/wandb"
    metadata = get_wandb_metadata(wandb_root_dir)
    print("metadata: ", metadata)
        
    run = get_wandb_run_until(wandb_root_dir)
    print("Run Summary: ", run["summary"])
    print("History Steps: ", len(run["history"]))
    print("Stats: ", len(run["stats"]))

    for step, data in run["history"].items():
        print(f"Step {step}: {data}")
        


if __name__ == "__main__":
    test_wandb()
