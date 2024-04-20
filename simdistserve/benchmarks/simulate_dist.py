"""
Simulate DistServe

Output a JSON (list) where each item is the lifecycle for a request.
"""
import argparse
import json
import random
from pathlib import Path
from typing import Literal, Union

import numpy as np
import pandas as pd
import simpy

from simdistserve.base.organize_data import organize_request_df, organize_request_event_df, \
    calculate_per_request_latency, organize_worker_event_df
from simdistserve.base.scheduler import put_requests_with_interarrivals
from simdistserve.base.worker import WorkerConfig
from simdistserve.base.workload import (
    get_gamma_interarrival,
    get_fixed_interarrival,
    convert_absolutearrival_to_interarrival, convert_pd_pair_to_request, sample_requests
)
from simdistserve.clusters.disagg import DisaggCluster
from simdistserve.clusters.vllm import VLLMCluster
from simdistserve.timemodule.worktime import ModelTypes


def parse_args(args_=None):
    parser = argparse.ArgumentParser(description='Simulation: vLLM, DistServe')
    parser.add_argument('--backend', type=str, default='distserve',
                        help='Backend to simulate (distserve, vllm)')
    parser.add_argument('--model', type=str, default='opt_13b',
                        help='Model type (opt_13b, opt_66b, opt_175b)')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--rate', type=float, default=float("inf"),
                        help='Rate of requests per second')
    parser.add_argument('--N', type=int, default=64, help='Number of requests')
    parser.add_argument(
        '--arrival', type=str, default='poisson',
        help=('Arrival distribution (gamma, poisson, fixed, custom). '
              'If custom, then require the JSON file workload to specify '
              'the "start_time" field for each incoming request.'))
    parser.add_argument(
        '--workload', type=str, default='sharegpt',
        help=(
            'Workload type, or a JSON file that contains the workload. '
            'The workload file should be a list of pairs with (prompt_len, decode_len) length. '
            '(e.g.: "sharegpt", "longbench", "humaneval", or specify your own path like "./workload/workload.json")')
    )
    parser.add_argument('--cv', type=float, default=1.0)
    parser.add_argument('--tp-prefill', type=int, default=1, help='Number of TP per prefill worker (used in DistServe)')
    parser.add_argument('--pp-prefill', type=int, default=1, help='Number of PP per prefill worker (used in DistServe)')
    parser.add_argument('--tp-decode', type=int, default=1, help='Number of TP per decode worker (used in DistServe)')
    parser.add_argument('--pp-decode', type=int, default=1, help='Number of PP per decode worker (used in DistServe)')
    parser.add_argument('--name', type=str, default=None)  # Experiment name
    parser.add_argument('--output', type=str, default=None, help='Output SLA (csv)')
    parser.add_argument('--output-request-info', type=str, default=None, help='Output request info (csv)')
    parser.add_argument('--output-request-event', type=str, default=None,
                        help='Output per-request event dataframe (csv)')
    parser.add_argument('--output-request-latency', type=str, default=None, help='Output per-request latency (csv)')
    parser.add_argument('--output-worker', type=str, default=None,
                        help='Output per-worker per-iteration time (csv)')
    parser.add_argument('--prefill-containment', type=int, default=None,
                        help='Containment target for prefill')
    parser.add_argument('--prefill-target', type=int, default=200,
                        help='Target latency for prefill')
    parser.add_argument('--decode-containment', type=int, default=None,
                        help='Containment target for decode')
    parser.add_argument('--slo-scales', type=str, default='[1.0, 0.4, 0.6, 0.8, 1.2]',
                        help='SLO scales in a python list.'),
    parser.add_argument('--decode-target', type=int, default=100,
                        help='Target latency for decode')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Print verbose output')

    args = parser.parse_args(args=args_)

    assert args.backend in ['distserve', 'vllm'], f'Unknown backend: {args.backend}'
    assert args.arrival in ['poisson', 'gamma', 'fixed', 'custom'], f'Unknown arrival process: {args.arrival}'
    args.slo_scales = eval(args.slo_scales)
    assert isinstance(args.slo_scales, list)
    return args


def main(args):
    cv = args.cv
    N = args.N
    rate = args.rate
    seed = args.seed
    workload: Union[Literal["sharegpt", "longbench", "humaneval"], str] = args.workload
    if args.model == 'opt_13b':
        model_type = ModelTypes.opt_13b
    elif args.model == 'opt_66b':
        model_type = ModelTypes.opt_66b
    elif args.model == 'opt_175b':
        model_type = ModelTypes.opt_175b
    else:
        raise ValueError(f'Unknown model type: {args.model}. Only support: ["opt_13b", "opt_66b", "opt_175b"].')

    PP_prefill = args.pp_prefill
    PP_decode = args.pp_decode
    TP_Prefill = args.tp_prefill
    TP_Decode = args.tp_decode

    # Setting the seed to sample request / process
    random.seed(args.seed)
    np.random.seed(args.seed)

    # TODO: Check the random process is set correctly.
    if workload in ['sharegpt', 'longbench', 'humaneval']:
        dataset_root = Path(__file__).parent
        if workload == 'sharegpt':
            requests = sample_requests(dataset_root / 'data/sharegpt.dataset', N)
        elif workload == 'longbench':
            requests = sample_requests(dataset_root / 'data/longbench.dataset', N)
        elif workload == 'humaneval':
            requests = sample_requests(dataset_root / 'data/humaneval.dataset', N)
        else:
            raise ValueError(
                f'Unknown workload type: {workload}. '
                f'Only support: ["sharegpt", "longbench", "humaneval"] or specified custom path to workload'
            )
        if args.arrival == 'fixed':
            delay = 1 / rate * 1000  # ms
            arrival = get_fixed_interarrival(N, delay)
        else:
            arrival = get_gamma_interarrival(N, rate, cv, seed=seed)

    else:
        # Open the file to get the JSON data
        # [ { "start_time": int, "prompt_len": int, "output_len":int,  } ]
        with open(workload, 'r') as f:
            data = json.load(f)
        request_pairs = [(d['prompt_len'], d['output_len']) for d in data]
        requests = convert_pd_pair_to_request(request_pairs)
        absolute_arrival = [d['start_time'] for d in data]
        arrival = convert_absolutearrival_to_interarrival(absolute_arrival)
        pass

    # Run simulation
    env = simpy.Environment()
    if args.backend == 'vllm':
        cluster = VLLMCluster(
            env=env,
            N_instance=1,
            PP=PP_prefill,
            worker_configs=WorkerConfig(
                model_type=model_type,
                TP=TP_Prefill, TP_Prefill=TP_Prefill, TP_Decode=TP_Prefill,
                chunked_prefill_max_tokens=0,  # TODO: Check what value this should be set to.
                prefill_max_batch_size=0,  # TODO: Set this as the max batch size.

            ),
        ).run()
        put_requests_with_interarrivals(env, cluster.scheduler, arrival, requests)
        env.run()
    elif args.backend == 'distserve':
        cluster = DisaggCluster(
            env=env,
            N_prefill_instance=1,
            N_decode_instance=1,
            PP_prefill=PP_prefill,
            PP_decode=PP_decode,
            worker_configs=WorkerConfig(
                model_type=model_type,
                TP_Prefill=TP_Prefill, TP_Decode=TP_Decode,
                TP=1,  # TODO(Refactor): Deprecated field, just use the default value in case of exception.
                chunked_prefill_max_tokens=0,  # TODO: Check what value this should be set to.
                prefill_max_batch_size=0,  # TODO: Set this as the max batch size.
            ),
        ).run()
        put_requests_with_interarrivals(env, cluster.scheduler, arrival, requests)
        env.run()

    #
    # Handle vllm in data processing
    #
    if args.backend == 'vllm':
        TP_Decode = PP_decode = 0

    #
    # Collect request-level data and containment
    #
    request_df = organize_request_df(requests)
    request_event_df = organize_request_event_df(requests)
    per_request_latency_df = calculate_per_request_latency(
        request_event_df, request_df.output_lens
    )
    if args.output_request_info:
        with open(args.output_request_info, 'w') as f:
            request_df.to_csv(f, index=False)
    if args.output_request_event:
        with open(args.output_request_event, 'w') as f:
            request_event_df.to_csv(f, index=False)
    if args.output_request_latency:
        with open(args.output_request_latency, 'w') as f:
            per_request_latency_df.to_csv(f, index=False)

    columns = [
        "backend", "model_type", "pd", "rate", "target", "attainment",
        "tp_prefill", "pp_prefill", "tp_decode", "pp_decode",
    ]
    output_results = []
    # Fix the prefill & decode target (SLO & scale),
    # then find the attainment (percentage of requests that meet the SLO)
    for scale in args.slo_scales:
        prefill_target = args.prefill_target * scale
        prefill_attainment = (per_request_latency_df['first_token_latency'] <= prefill_target).sum() / N
        prefill_attainment *= 100
        item = [args.backend, model_type, 'prefill', rate, prefill_target, prefill_attainment,
                TP_Prefill, PP_prefill, TP_Decode, PP_decode]
        output_results.append(item)

        decode_target = args.decode_target * scale
        decode_attainment = (per_request_latency_df['tpot'] <= decode_target).sum() / N
        decode_attainment *= 100
        item = [args.backend, model_type, 'decode', rate, decode_target, decode_attainment,
                TP_Prefill, PP_prefill, TP_Decode, PP_decode]
        output_results.append(item)

        both_attainment = (
                              (per_request_latency_df['first_token_latency'] <= prefill_target) &
                              (per_request_latency_df['tpot'] <= decode_target)
                          ).sum() / N
        both_attainment *= 100
        item = [args.backend, model_type, 'both', rate, (prefill_target, decode_target), both_attainment,
                TP_Prefill, PP_prefill, TP_Decode, PP_decode]
        output_results.append(item)

        pass

    # Fix the attainment (percentage of requests that meet the SLO),
    # then find the prefill /  decode SLO target that it can meet.
    slas = [85, 90, 95, 98, 99]
    for sla in slas:
        prefill_attainment = decode_attainment = sla
        prefill_target = per_request_latency_df['first_token_latency'].quantile(prefill_attainment / 100)
        decode_target = per_request_latency_df['tpot'].quantile(decode_attainment / 100)
        item = [args.backend, model_type, 'prefill', rate, prefill_target, prefill_attainment,
                TP_Prefill, PP_prefill, TP_Decode, PP_decode]
        output_results.append(item)
        item = [args.backend, model_type, 'decode', rate, decode_target, decode_attainment,
                TP_Prefill, PP_prefill, TP_Decode, PP_decode]
        output_results.append(item)
        pass

    df = pd.DataFrame(output_results, columns=columns)

    if args.output:
        with open(args.output, 'w') as f:
            df.to_csv(f, index=False)

    if args.verbose:
        print(df.to_markdown())

    #
    # Collect worker-level data
    #
    if args.output_worker:
        worker_df = organize_worker_event_df(cluster)
        worker_df.to_csv(args.output_worker, index=False)

    #
    # Return if the agreement of prefill/decode is met
    #
    is_prefill_contained = None
    is_decode_contained = None

    prefill_containment = args.prefill_containment
    prefill_target = args.prefill_target
    if prefill_containment:
        # See if the P{prefill_containment} is less than prefill_target
        t = per_request_latency_df['first_token_latency'].quantile(prefill_containment / 100)
        is_prefill_contained = t < prefill_target
        pass

    decode_containment = args.decode_containment
    decode_target = args.decode_target
    if decode_containment:
        t = per_request_latency_df['tpot'].quantile(decode_containment / 100)
        is_decode_contained = t < decode_target
        pass

    return is_prefill_contained, is_decode_contained, df


def test_opt_13b_example():
    arg_lists = [
        [
            '--arrival', 'poisson',
            '--seed', '0',
            '--N', '1000',
            '--prefill-containment', '90',  # P90
            '--prefill-target', '200',  # ms
            '--decode-containment', '90',  # P90
            '--decode-target', '100',  # ms
            '--model', 'opt_13b',
            '--workload', 'sharegpt',
        ]
    ]

    config_list = [
        [
            '--rate', f'{rate}',
            '--output',
            f'raw_results/request.opt-13b-p{tp_prefill}{pp_prefill}{tp_decode}{pp_decode}-rate{rate}.csv',
            '--output-worker',
            f'raw_results/worker.opt-13b-p{tp_prefill}{pp_prefill}{tp_decode}{pp_decode}-rate{rate}.csv',
            '--pp-prefill', f'{pp_prefill}',
            '--pp-decode', f'{pp_decode}',
            '--tp-prefill', f'{tp_prefill}',
            '--tp-decode', f'{tp_decode}',
        ]
        for rate in range(1, 50)
        for pp_prefill in [1, 2, 4, 8]
        for pp_decode in [1, 2, 4, 8]
        for tp_prefill in [1, 2, 4, 8]
        for tp_decode in [1, 2, 4, 8]
    ]

    best_config = None
    best_goodput = 0
    # pbar = tqdm(total=len(arg_lists) * len(config_list))

    print("tp_prefill,pp_prefill,tp_decode,pp_decode,rate,goodput")
    for machine_config in config_list:
        best_config_this_iter = None
        for task_config in arg_lists:
            # pbar.update(1)

            args = parse_args(args_=task_config + machine_config)
            key = (
                args.tp_prefill, args.pp_prefill,
                args.tp_decode, args.pp_decode,
            )

            rate = args.rate
            num_gpu = args.pp_prefill * args.tp_prefill + args.pp_decode * args.tp_decode
            if num_gpu > 32:
                continue
            goodput = args.rate / num_gpu
            if goodput < best_goodput:
                continue

            # print(args.rate, args.pp_prefill, args.tp_prefill, args.pp_decode, args.tp_decode)
            is_prefill_contained, is_decode_contained, containment_df = main(args)

            if not is_prefill_contained or not is_decode_contained:
                break

            if goodput > best_goodput:
                best_config = args
                best_goodput = goodput

                print(
                    f"{best_config.tp_prefill},{best_config.pp_prefill},"
                    f"{best_config.tp_decode},{best_config.pp_decode},"
                    f"{rate},{best_goodput}")

    print(f"Best Config: {best_config} with goodput {best_goodput}")


if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)
    pass