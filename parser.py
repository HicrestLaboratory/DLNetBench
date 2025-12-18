import json
import pandas as pd
import matplotlib.pyplot as plt
import sbatchman as sbm
import sys
import numpy as np
import os
from argparse import ArgumentParser

# Make sure Python can find ccutils
home = os.getenv("HOME")
sys.path.append(f"{home}")

from ccutils.parser.ccutils_parser import *

def extract_fsdp_metrics_df(fsdp_section, job_vars):
    """
    Extract FSDP metrics from a Section object and job variables into two Pandas DataFrames:
      - runtime_df: one row per rank per run
      - comm_df: one row per rank per run per unit with both allgather and reduce_scatter times
    """
    global_data = fsdp_section.json_data

    # Global metrics
    world_size = job_vars.get("nodes")
    network_type = job_vars.get("partition", "unknown")
    model_name = job_vars.get("models_fsdp")
    
    global_params = {
        "network": network_type,
        "world_size": world_size,
        "sharding_factor": global_data.get("sharding_factor"),
        "num_replicas": global_data.get("num_replicas"),
        "model_name": model_name,
        "model_size_bytes": global_data.get("model_size_bytes"),
        "local_batch_size": global_data.get("local_batch_size"),
        "num_units": global_data.get("num_units"),
        "fwd_time_per_unit_us": global_data.get("fwd_time_per_unit_us"),
        "bwd_time_per_unit_us": global_data.get("bwd_time_per_unit_us"),
        "allgather_msg_size_bytes": global_data.get("allgather_msg_size_bytes"),
        "reducescatter_msg_size_bytes": global_data.get("reducescatter_msg_size_bytes"),
    }

    runtime_rows = []
    comm_rows = []

    for rank, json_str in fsdp_section.mpi_all_prints["ccutils_rank_json"].rank_outputs.items():
        parsed = json.loads(json_str)
        runtimes = parsed.get("runtime", [])
        barriers = parsed.get("barrier", [])
        allgather_times = parsed.get("allgather", [])
        reduce_scatter_times = parsed.get("reduce_scatter", [])
        
        num_runs = len(runtimes)
        num_units = global_params["num_units"] or max(len(reduce_scatter_times)//num_runs, 1)

        # Build runtime DataFrame
        for run_idx in range(num_runs):
            runtime_rows.append({
                **global_params,
                "rank": rank,
                "run": run_idx,
                "runtime": runtimes[run_idx],
                "barrier": barriers[run_idx] if run_idx < len(barriers) else 0
            })

        # Build communication DataFrame with both times
        for run_idx in range(num_runs):
            for unit_idx in range(num_units):
                ag_idx = run_idx * 2 * num_units + unit_idx*2
                rs_idx = run_idx * num_units + unit_idx

                comm_rows.append({
                    **global_params,
                    "rank": rank,
                    "run": run_idx,
                    "unit_idx": unit_idx,
                    "allgather": allgather_times[ag_idx] if ag_idx < len(allgather_times) else 0,
                    "reduce_scatter": reduce_scatter_times[rs_idx] if rs_idx < len(reduce_scatter_times) else 0
                })

    runtime_df = pd.DataFrame(runtime_rows)
    comm_df = pd.DataFrame(comm_rows)
    return runtime_df, comm_df



def extract_dp_metrics_df(dp_section, job_vars):
    """
    Extract DP metrics from a Section object and job variables into a Pandas DataFrame.
    Returns a DataFrame with one row per rank measurement.
    """
    json_data = dp_section.json_data

    world_size = job_vars.get("nodes", json_data.get("world_size"))
    network_type = job_vars.get("partition", "unknown")
    model_name = job_vars.get("models", json_data.get("model_name"))
    local_batch_size = json_data.get("local_batch_size")
    num_buckets = job_vars.get("num_buckets", json_data.get("num_buckets"))
    fwd_rt = json_data.get("fwd_rt_whole_model_s")
    bwd_rt = json_data.get("bwd_rt_per_bucket_s")
    msg_avg = json_data.get("msg_size_avg_bytes")
    msg_std = json_data.get("msg_size_std_bytes")

    rows = []
    rank_outputs = dp_section.mpi_all_prints["ccutils_rank_json"].rank_outputs

    for rank, json_str in rank_outputs.items():
        parsed = json.loads(json_str)
        runtimes = parsed["runtimes"]
        barrier_times = parsed["barrier_time_us"]

        for rt, bt in zip(runtimes, barrier_times):
            row = {
                "network": network_type,
                "world_size": world_size,
                "model_name": model_name,
                "local_batch_size": local_batch_size,
                "num_buckets": num_buckets,
                "fwd_rt_whole_model_s": fwd_rt,
                "bwd_rt_per_bucket_s": bwd_rt,
                "msg_size_avg_bytes": msg_avg,
                "msg_size_std_bytes": msg_std,
                "rank": rank,
                "runtime_s": rt,
                "barrier_time_s": bt
            }
            rows.append(row)

    return pd.DataFrame(rows)


def extract_metrics_df(strategy: str):
    """
    Returns the correct extraction function depending on the strategy.
    FSDP returns a tuple of (runtime_df, comm_df)
    DP returns a single DataFrame.
    """
    if strategy == "fsdp":
        return extract_fsdp_metrics_df
    elif strategy == "dp":
        return extract_dp_metrics_df
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def get_metrics_dataframe(strategy: str = "dp"):
    """
    Process all completed jobs and return a combined Pandas DataFrame
    with metrics for all ranks.
    
    For FSDP, returns a tuple: (runtime_df, comm_df)
    For DP, returns a single DataFrame.
    """
    jobs = sbm.jobs_list(status=[sbm.Status.COMPLETED])
    all_dfs = []

    for job in jobs:
        job_output = job.get_stdout()
        mpi_parser = MPIOutputParser()
        parser_output = mpi_parser.parse_string(job_output)

        section = parser_output.get(strategy)
        if section:
            job_vars = job.variables
            df_or_tuple = extract_metrics_df(strategy)(section, job_vars)
            all_dfs.append(df_or_tuple)

    if not all_dfs:
        if strategy == "fsdp":
            return pd.DataFrame(), pd.DataFrame()
        else:
            return pd.DataFrame()

    if strategy == "fsdp":
        # Concatenate runtime and comm separately
        runtime_dfs, comm_dfs = zip(*all_dfs)
        full_runtime_df = pd.concat(runtime_dfs, ignore_index=True)
        full_comm_df = pd.concat(comm_dfs, ignore_index=True)
        return full_runtime_df, full_comm_df
    else:
        # DP: just concatenate the single DataFrames
        full_df = pd.concat(all_dfs, ignore_index=True)
        return full_df