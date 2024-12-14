#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os, sys
from matplotlib import pyplot as plt
import dataclasses
from typing import Optional

# sys.path.append(os.path.join(os.getcwd(), "..", "..", "simulator"))
from structs import Dataset, read_request_results, RequestResult

@dataclasses.dataclass
class Backend:
    name: str
    label: str
    color: str
    num_gpus: int

EXP_RESULT_ROOT="/projects/bdes/kpatel11/DistServe/evaluation/docs/results"
SYSNAME="DistServe"

def load_result(exp_result_dir: str, backend: Backend, per_gpu_num_prompt: int, per_gpu_request_rate: float) -> list[RequestResult]:
    possible_paths = [
        f"{EXP_RESULT_ROOT}/{exp_result_dir}/{backend.name}-{per_gpu_num_prompt*backend.num_gpus}-{per_gpu_request_rate*backend.num_gpus}.exp",
        f"{EXP_RESULT_ROOT}/{exp_result_dir}/{backend.name}-{per_gpu_num_prompt*backend.num_gpus}-{round(per_gpu_request_rate*backend.num_gpus, 4)}.exp"
    ]
    if int(per_gpu_request_rate*backend.num_gpus) == per_gpu_request_rate*backend.num_gpus:
        possible_paths.append(f"{EXP_RESULT_ROOT}/{exp_result_dir}/{backend.name}-{per_gpu_num_prompt*backend.num_gpus}-{int(per_gpu_request_rate*backend.num_gpus)}.exp")
    for path in possible_paths:
        if os.path.exists(path):
            return read_request_results(path)
    raise FileNotFoundError(f"Cannot find result file for {backend.name}, {per_gpu_num_prompt}, {per_gpu_request_rate} (filename candidates: {possible_paths})")
        

def get_attainment(results: list[RequestResult], ttft_slo: Optional[float], tpot_slo: Optional[float]):
    if ttft_slo is None: ttft_slo = 1e10
    if tpot_slo is None: tpot_slo = 1e10
    counter = 0
    for result in results:
        if result.ftl <= ttft_slo and result.tpot <= tpot_slo:
            counter += 1
    return (counter / len(results))*100

def save_plot_to(filename: str):
    path = os.path.join("/projects/bdes/kpatel11/DistServe/evaluation/docs/plots", filename)
    plt.savefig(path, bbox_inches="tight")

def get_marker(backend: Backend):
    if backend.name == SYSNAME:
        return "o"
    elif "vllm" in backend.name:
        return "v"
    elif "deepspeed" in backend.name.lower():
        return "D"
    else:
        assert False, f"Unknown backend: {backend.name}"



from distserve.lifetime import LifetimeEvent, LifetimeEventType, json_decode_lifetime_events
import numpy as np

stage_to_strs: list[str] = [
    "Prefill Queuing",
    "Prefill Execution",
    "Transmission",
    "Decoding Queuing",
    "Decoding Execution"
]
num_stages = len(stage_to_strs)
    
def analyse_one_request_result(result: RequestResult) -> np.array:
    """
    analyse_one_request_result: Analyse the time spend on a single request.
    """
    lifetime_event_dict = {}    # LifetimeEventType -> timestamp
    for event in result.lifecycle_events:
        lifetime_event_dict[event.event_type] = event.timestamp
    
    def get_event_timestamp_delta(event_type1: LifetimeEventType, event_type2: LifetimeEventType) -> float:
        return lifetime_event_dict[event_type2] - lifetime_event_dict[event_type1]
    
    try:
        result = np.array([
            get_event_timestamp_delta(LifetimeEventType.Issued, LifetimeEventType.ContextBegin),
            get_event_timestamp_delta(LifetimeEventType.ContextBegin, LifetimeEventType.ContextEnd),
            get_event_timestamp_delta(LifetimeEventType.MigrationBegin, LifetimeEventType.MigrationEnd),
            get_event_timestamp_delta(LifetimeEventType.ContextEnd, LifetimeEventType.MigrationBegin) + get_event_timestamp_delta(LifetimeEventType.MigrationEnd, LifetimeEventType.DecodingBegin),
            get_event_timestamp_delta(LifetimeEventType.DecodingBegin, LifetimeEventType.DecodingEnd)
        ])
    except:
        result = np.array([])

    return result

def analyse_request_results(request_results: list[RequestResult]) -> np.array:
    """
    analyse_request_results: Analyse the time spend on all the requests and get an average of every stage's time spend fraction.
    """
    num_requests = len(request_results)
    total_time_spend_fracs = np.zeros(num_stages)
    for result in request_results:
        cur_result = analyse_one_request_result(result)
        if len(cur_result) == 0:
            continue
        cur_result /= cur_result.sum()
        cur_result *= 100
        total_time_spend_fracs += cur_result
    total_time_spend_fracs /= num_requests
    return total_time_spend_fracs

def plot_microbenchmark_bar(
    ax: plt.Axes,
    exp_result_dir: str,
    num_prompts_req_rates: list[(int, float)],
    backend: Backend
):
    colors = ["lightgreen", "#5b5", "red", "lightskyblue", "deepskyblue"]
    bar_width = 0.5
    plt.rcParams.update({'font.size': 14})
    bottoms = np.zeros(len(num_prompts_req_rates))
    for index in range(num_stages):
        stage_str = stage_to_strs[index]
        stage_time_fracs = []
        for num_prompts, req_rate in num_prompts_req_rates:
            results = load_result(exp_result_dir, backend, num_prompts, req_rate)
            time_frac = analyse_request_results(results)[index]
            stage_time_fracs.append(time_frac)
        ax.bar(
            [f"{round(req_rate, 2)}" for num_prompts, req_rate in num_prompts_req_rates],
            stage_time_fracs,
            bar_width,
            label=stage_str,
            bottom=bottoms,
            color=colors[index]
        )
        bottoms += stage_time_fracs

    ax.set_xlabel("RPS")
    ax.set_ylabel("Latency Breakdown (%)")
    ax.legend(frameon=False, loc = (-0.12, 1.0), ncol=2,
          columnspacing=0.5)

'''
def plot_migration_time_usage_cdf(
    ax: plt.Axes,
    label: str,
    exp_result_dir: str,
    num_prompts: int,
    request_rate: float,
    backend: Backend
):
    ax.set_ylabel("CDF")
    ax.set_xlabel("Transmission Time (s)")
    req_results = load_result(exp_result_dir, backend, num_prompts, request_rate)
    transmission_times = [
        analyse_one_request_result(result)[2]
        for result in req_results   
    ]
    transmission_times.sort()
    ax.ecdf(transmission_times, label=label)
'''

def plot_fig10():
    plt.rcParams.update({'font.size': 14})
    # fig, axs = plt.subplots(1, 2, figsize=(10, 3))
    fig, axs = plt.subplots(1,1, figsize=(5,3))
    plot_microbenchmark_bar(
        axs,
        "opt-13b-priority-sjf",
        [(100, 0.5), (100, 1), (100, 1.5), (100, 2), (100, 2.5), (100, 3)],
        Backend("distserve", SYSNAME, "C0", 1)
    )
    '''
    plot_migration_time_usage_cdf(
        axs[1],
        "OPT-13B",
        "opt-13b-sharegpt",
        100, 2,
        Backend("distserve", SYSNAME, "C0", 3)
    )
    plot_migration_time_usage_cdf(
        axs[1],
        "OPT-66B",
        "opt-66b-sharegpt",
        25, 0.375,
        Backend("distserve", SYSNAME, "C0", 8)
    )
    plot_migration_time_usage_cdf(
        axs[1],
        "OPT-175B",
        "opt-175b-sharegpt",
        25, 0.1875,
        Backend("distserve", SYSNAME, "C0", 21)
    )
    axs[1].legend(frameon=False, loc="lower right")
    '''
'''
plot_fig10()
save_plot_to("microbenchmark-sjf.pdf")
plt.show()
'''


def analyse_request_results_no_frac(request_results: list[RequestResult]) -> np.array:
    """
    analyse_request_results: Analyse the time spend on all the requests and get an average of every stage's time spend fraction.
    """
    num_requests = len(request_results)
    total_time_spend = np.zeros(num_stages)
    for result in request_results:
        cur_result = analyse_one_request_result(result)
        if len(cur_result) == 0:
            continue
        total_time_spend += cur_result
    total_time_spend /= num_requests
    return total_time_spend



def plot_microbenchmark_sched_comparison(
    exp_result_dirs: list[str],
    num_prompts_req_rates: list[(int, float)],
    plotting_metric: str,
    backend: Backend
):
    colors = ["lightgreen", "#5b5", "red", "lightskyblue", "deepskyblue"]
    bar_width = 0.5
    plt.rcParams.update({'font.size': 14})
    bottoms = np.zeros(len(num_prompts_req_rates))

    for num_prompts, req_rate in num_prompts_req_rates:
        stage_time_split = []

        for index in range(num_stages):
            stage_str = stage_to_strs[index]
            if plotting_metric == "queuing":
                if stage_str != 'Prefill Queuing' and stage_str != 'Decoding Queuing':
                    continue
            elif plotting_metric == "execution":
                if stage_str != 'Prefill Execution' and stage_str != 'Decoding Execution':
                    continue
            sched_policy_time_split = []

            for exp_result_dir in exp_result_dirs:
                results = load_result(exp_result_dir, backend, num_prompts, req_rate)
                time_split = analyse_request_results_no_frac(results)[index]
                sched_policy_time_split.append(time_split)

            stage_time_split.append(sched_policy_time_split)

        plt.rcParams.update({'font.size': 10})
        fig, ax =  plt.subplots(1,1, figsize=(5,3))

        num_policies = len(exp_result_dirs)
        values = list(zip(*stage_time_split))

        n_groups = len(stage_time_split)
        index = np.arange(n_groups)
        bar_width = 0.8 / num_policies
        colors = plt.cm.viridis(np.linspace(0, 1, num_policies))

        # Plot bars for each policy
        for i, policy_values in enumerate(values):
            ax.bar(index + i * bar_width, policy_values, bar_width, label=exp_result_dirs[i], color=colors[i])

        # Adding labels, title, and legend
        ax.set_xlabel('Stages')
        ax.set_ylabel('Latency')
        ax.set_xticks(index + bar_width * (num_policies - 1) / 2)
        if plotting_metric == "queuing":
            ax.set_xticklabels(['Prefill Queuing', 'Decode Queuing'])
        elif plotting_metric == "execution":
            ax.set_xticklabels(['Prefill Execution', 'Decode Execution'])
        ax.legend(fontsize=5)
        fig.tight_layout()
        plt.savefig(f'../docs/plots/policy_comparison/queueing_time_comp_({num_prompts},{req_rate}).png', dpi=400)


def plot_fig11():
    # fig, axs = plt.subplots(1, 2, figsize=(10, 3))
    plot_microbenchmark_sched_comparison(
        ["opt-13b-fcfs-context-decode-1111", "opt-13b-priority-bounded-sjf-context-decode-1111"],
        [
            # (100, 0.5), 
            (500, 10)
        ],
        "queuing",
        Backend("distserve", SYSNAME, "C0", 1)
    )

# plot_fig11()


fcfs_output_len = []


def get_request_total_time_server(request_results: list[RequestResult], exp_result_dir: str) -> np.array:
    """
    analyse_request_results: Analyse the time spend on all the requests and get an average of every stage's time spend fraction.
    """
    ouput_len_to_time_spent = []
    for i,result in enumerate(request_results):
        cur_result = analyse_one_request_result(result)
        if len(cur_result) == 0:
            continue
        ouput_len_to_time_spent.append((result.output_len, sum(cur_result)))
    return ouput_len_to_time_spent


def get_request_total_time_client(request_results: list[RequestResult], exp_result_dir: str) -> np.array:
    """
    analyse_request_results: Analyse the time spend on all the requests and get an average of every stage's time spend fraction.
    """
    ouput_len_to_time_spent = []
    for i, result in enumerate(request_results):
        cur_result = analyse_one_request_result(result)
        if len(cur_result) == 0:
            continue
        ouput_len_to_time_spent.append((result.output_len, result.end_time - result.start_time))
    return ouput_len_to_time_spent


def plot_individual_request_latency(
    exp_result_dirs: list[str],
    num_prompts_req_rates: list[tuple[int, float]],
    view: str,
    backend: Backend
):

    plt.rcParams.update({'font.size': 14})

    for num_prompts, req_rate in num_prompts_req_rates:
        total_time = []

        for exp_result_dir in exp_result_dirs:
            results = load_result(exp_result_dir, backend, num_prompts, req_rate)
            if view == "server":
                ouput_len_to_time_spent = get_request_total_time_server(results, exp_result_dir)
            elif view == "client":
                ouput_len_to_time_spent = get_request_total_time_client(results, exp_result_dir)

            ouput_len_to_time_spent = sorted(ouput_len_to_time_spent, key=lambda x: x[0])
            total_time.append(np.array([x[1] for x in ouput_len_to_time_spent]))

            print(ouput_len_to_time_spent)
            print("\n")


        plt.rcParams.update({'font.size': 10})
        fig, ax =  plt.subplots(1,1, figsize=(8,3))

        print(total_time[0] - total_time[1])
        print(sum(total_time[0] - total_time[1]))
        ax.plot( np.arange(len(total_time[0])), total_time[0] - total_time[1] )
            
        ax.set_xlabel("Requests")
        ax.set_ylabel("Delta b/w FCFS and SJF Latency per Request")
        fig.tight_layout()
        plt.savefig(f'../docs/plots/policy_comparison/delta_request_latency_per_request_{view}_side_({num_prompts},{req_rate}).png', dpi=400)

        '''
        plt.rcParams.update({'font.size': 10})
        fig, ax =  plt.subplots(1,1, figsize=(8,3))

        for i,exp_result_dir in enumerate(exp_result_dirs):
            ax.plot( np.arange(len(total_time[i])), total_time[i], label=exp_result_dir )
            
        ax.set_xlabel("Requests")
        ax.set_ylabel("Latency")
        ax.legend(fontsize=8)
        fig.tight_layout()
        plt.savefig(f'../docs/plots/policy_comparison/request_latency_per_request_({num_prompts},{req_rate}).png', dpi=400)
        '''


def plot_fig12():
    # fig, axs = plt.subplots(1, 2, figsize=(10, 3))
    plot_individual_request_latency(
        ["opt-13b-fcfs-context-fcfs-decode-batch1-1111", "opt-13b-priority-sjf-context-priority-sjf-decode-batch1-1111"],
        [
            (50, 5)
        ],
        "server",
        Backend("distserve", SYSNAME, "C0", 1)
    )

plot_fig12()




def get_stage_start_time(request_results: list[RequestResult], stage: str) -> np.array:
    """
    analyse_request_results: Analyse the time spend on all the requests and get an average of every stage's time spend fraction.
    """
    ouput_len_to_time_spent = []
    for result in request_results:
        for event in result.lifecycle_events:
            if stage == "context":
                if event.event_type == LifetimeEventType.ContextBegin:
                    ouput_len_to_time_spent.append((result.output_len, event.timestamp))
            elif stage == "decode":
                if event.event_type == LifetimeEventType.DecodingBegin:
                    ouput_len_to_time_spent.append((result.output_len, event.timestamp))
    return ouput_len_to_time_spent


def plot_context_begin_timestamp(
    exp_result_dir: str,
    num_prompts_req_rates: list[tuple[int, float]],
    stage: str,
    backend: Backend
):

    plt.rcParams.update({'font.size': 14})

    for num_prompts, req_rate in num_prompts_req_rates:

        results = load_result(exp_result_dir, backend, num_prompts, req_rate)
        output_len_to_time_spent = get_stage_start_time(results, stage)

        output_len_to_time_spent = sorted(output_len_to_time_spent, key=lambda x: x[1])


        plt.rcParams.update({'font.size': 10})
        fig, ax =  plt.subplots(1,1, figsize=(8,3))

        ax.plot( [x[1] for x in output_len_to_time_spent], [x[0] for x in output_len_to_time_spent] )
            
        ax.set_xlabel(f"{stage} Start Time")
        ax.set_ylabel("Output Length")
        fig.tight_layout()
        plt.savefig(f'../docs/plots/policy_comparison/fcfs_batch1_{stage}_start_v_output_len_({num_prompts},{req_rate}).png', dpi=400)



def plot_fig13():
    # fig, axs = plt.subplots(1, 2, figsize=(10, 3))
    plot_context_begin_timestamp(
        "opt-13b-fcfs-context-fcfs-decode-batch1-1111",
        [
            (50, 5)
        ],
        "context",
        Backend("distserve", SYSNAME, "C0", 1)
    )

plot_fig13()

# %%
