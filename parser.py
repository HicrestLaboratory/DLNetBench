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
    Extract FSDP metrics from a Section object and job variables into a Pandas DataFrame.
    Returns a DataFrame with one row per rank per run.
    Each run has: 1 runtime, 2*num_units allgather times, and num_units reduce_scatter times.
    """
    # Get global JSON data
    global_json = fsdp_section.mpi_all_prints.get("ccutils_global_json")
    if global_json:
        global_data = json.loads(global_json.rank_outputs.get(0, "{}"))
    else:
        global_data = {}
    
    # Extract global metrics
    world_size = job_vars.get("nodes", global_data.get("sharding_factor"))
    network_type = job_vars.get("partition", "unknown")
    model_name = job_vars.get("models", global_data.get("model_name"))
    local_batch_size = global_data.get("local_batch_size")
    num_units = global_data.get("num_units")
    fwd_time_per_unit_us = global_data.get("fwd_time_per_unit_us")
    bwd_time_per_unit_us = global_data.get("bwd_time_per_unit_us")
    allgather_msg_size_bytes = global_data.get("allgather_msg_size_bytes")
    reducescatter_msg_size_bytes = global_data.get("reducescatter_msg_size_bytes")
    model_size_bytes = global_data.get("model_size_bytes")
    sharding_factor = global_data.get("sharding_factor")
    num_replicas = global_data.get("num_replicas")
    
    rows = []
    rank_outputs = fsdp_section.mpi_all_prints["ccutils_rank_json"].rank_outputs
    
    for rank, json_str in rank_outputs.items():
        parsed = json.loads(json_str)
        runtimes = parsed.get("runtime", [])
        allgather_times = parsed.get("allgather", [])
        reduce_scatter_times = parsed.get("reduce_scatter", [])
        
        num_runs = len(runtimes)
        
        # For each run, extract the corresponding communication times
        for run_idx in range(num_runs):
            # Calculate indices for this run's communication times
            # Each run has 2*num_units allgathers and num_units reduce_scatters
            allgather_start = run_idx * 2 * num_units
            allgather_end = allgather_start + 2 * num_units
            reduce_scatter_start = run_idx * num_units
            reduce_scatter_end = reduce_scatter_start + num_units
            
            # Extract times for this run
            run_allgathers = allgather_times[allgather_start:allgather_end] if allgather_start < len(allgather_times) else []
            run_reduce_scatters = reduce_scatter_times[reduce_scatter_start:reduce_scatter_end] if reduce_scatter_start < len(reduce_scatter_times) else []
            
            row = {
                "network": network_type,
                "world_size": world_size,
                "sharding_factor": sharding_factor,
                "num_replicas": num_replicas,
                "model_name": model_name,
                "model_size_bytes": model_size_bytes,
                "local_batch_size": local_batch_size,
                "num_units": num_units,
                "fwd_time_per_unit_us": fwd_time_per_unit_us,
                "bwd_time_per_unit_us": bwd_time_per_unit_us,
                "allgather_msg_size_bytes": allgather_msg_size_bytes,
                "reducescatter_msg_size_bytes": reducescatter_msg_size_bytes,
                "rank": rank,
                "run": run_idx,
                "runtime_s": runtimes[run_idx],
                "allgather_times": run_allgathers,
                "reduce_scatter_times": run_reduce_scatters
            }
            rows.append(row)
    
    return pd.DataFrame(rows)


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


def extract_metrics_df(strategy:str):
    if strategy == "fsdp":
        return extract_fsdp_metrics_df
    elif strategy == "dp":
        return extract_dp_metrics_df
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def get_dp_dataframe(strategy:str="dp"):
    """
    Process all completed jobs and return a combined Pandas DataFrame
    with DP metrics for all ranks.
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
            df = extract_metrics_df(strategy)(section, job_vars)
            all_dfs.append(df)

    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)
    else:
        full_df = pd.DataFrame()

    return full_df