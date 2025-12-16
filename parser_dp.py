import json
import pandas as pd
import matplotlib.pyplot as plt
import sbatchman as sbm
import sys
import numpy as np
import os

# Make sure Python can find ccutils
home = os.getenv("HOME")
sys.path.append(f"{home}")

from ccutils.parser.ccutils_parser import *

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


def get_dp_dataframe():
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

        dp_section = parser_output.get("dp")
        if dp_section:
            job_vars = job.variables
            df = extract_dp_metrics_df(dp_section, job_vars)
            all_dfs.append(df)

    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)
    else:
        full_df = pd.DataFrame()

    return full_df