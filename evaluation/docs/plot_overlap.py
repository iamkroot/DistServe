import json
import matplotlib.pyplot as plt
import numpy as np
import random
import re

def plot_request_intervals(data, path):
    waiting_times = []
    decode_waiting_times = []

    # data = random.sample(data, 100)

    min_time = sorted(data, key=lambda x: x['start_time'])[0]['lifecycle_events'][0]['timestamp']
    max_time = sorted(data, key=lambda x: x['end_time'])[-1]['lifecycle_events'][6]['timestamp']
    # print("Total duration", max_time - min_time)

    sorted_data = sorted(data, key=lambda x: x['start_time']) # fcfs
    # sorted_data = sorted(data, key=lambda x: x['output_len']) # sjf
    
    plt.figure(figsize=(18, 12))
    
    color_map = plt.cm.get_cmap('tab10')  # You can change 'tab10' to other colormaps like 'Set1', 'Set2', etc.

    for i, request in enumerate(sorted_data):
        # Plot start and end times as a horizontal line
        color = color_map(i % color_map.N)

        events = request['lifecycle_events']
        # print(events)
        issued = events[0]['timestamp'] - min_time
        # print("issued: ", issued)

        prefill_start = events[1]['timestamp'] - min_time
        prefill_end = events[2]['timestamp'] - min_time

        if len(events) == 7:
            decode_start = events[5]['timestamp'] - min_time
            decode_end = events[6]['timestamp'] - min_time
        else:
            decode_start = None
            decode_end = None

        waiting_times.append(prefill_start - issued)
        decode_waiting_times.append(decode_start - prefill_end)
        
        plt.plot([issued], 
                 [i], 
                 marker='o',
                color=color,
                 linewidth=2)

        plt.plot([prefill_start, prefill_end], 
                 [i, i], 
                 marker='o',
                 color=color,
                 linewidth=2)

        if decode_start is not None:
            plt.plot([decode_start, decode_end], 
                    [i, i], 
                    marker='o',
                    color=color,
                    linewidth=2)
    
    waiting_times = sorted(waiting_times)
    avg_waiting_time = sum(waiting_times) / len(sorted_data)
    top1_waiting_time = waiting_times[len(sorted_data)- 1]
    top5_waiting_time = waiting_times[len(sorted_data) - 5]
    p90_waiting_time = waiting_times[int(0.90 * len(sorted_data))]

    regex = r"opt-13b-(.*?)-context-(.*?)-decode"
    match = re.search(regex, path)

    phase1, phase2 = match.groups()

    print(path)
    print("Context: ", phase1)
    print("Decode: ", phase2)

    print("top1 waiting time (prefill start - issued) = {:.3f}".format(top1_waiting_time))
    print("top5 waiting time (prefill start - issued) = {:.3f}".format(top5_waiting_time))
    print("p90 waiting time (prefill start - issued) = {:.3f}".format(p90_waiting_time))
    print("Avg waiting time (prefill start - issued) = {:.3f}".format(avg_waiting_time))
    print("")

    waiting_times = sorted(decode_waiting_times)
    avg_waiting_time = sum(decode_waiting_times) / len(sorted_data)
    top1_waiting_time = decode_waiting_times[len(sorted_data)- 1]
    top5_waiting_time = decode_waiting_times[len(sorted_data) - 5]
    p90_waiting_time = decode_waiting_times[int(0.90 * len(sorted_data))]

    print("top1 waiting time (decode end - prefill start) = {:.3f}".format(top1_waiting_time))
    print("top5 waiting time (decode end - prefill start) = {:.3f}".format(top5_waiting_time))
    print("p90 waiting time (decode end - prefill start) = {:.3f}".format(p90_waiting_time))
    print("Avg waiting time (decode end - prefill start) = {:.3f}".format(avg_waiting_time))
    print("")

    total_waiting_times = [waiting_times[i] + decode_waiting_times[i] for i in range(len(waiting_times))]
    avg_waiting_time = sum(total_waiting_times) / len(sorted_data)
    top1_waiting_time = total_waiting_times[len(sorted_data)- 1]
    top5_waiting_time = total_waiting_times[len(sorted_data) - 5]
    p90_waiting_time = total_waiting_times[int(0.90 * len(sorted_data))]

    print("top1 waiting time (total delay) = {:.3f}".format(top1_waiting_time))
    print("top5 waiting time (total delay) = {:.3f}".format(top5_waiting_time))
    print("p90 waiting time (total delay) = {:.3f}".format(p90_waiting_time))
    print("Avg waiting time (total delay) = {:.3f}".format(avg_waiting_time))
    print("")
    
    plt.title('Request Intervals', fontsize=15)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Requests', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Improve y-axis ticks
    # plt.yticks(range(len(sorted_data)), range(len(sorted_data)))
    
    plt.tight_layout()
    save_path = path.rstrip('.exp') + "-overlap.png"

    # plt.savefig('./results/opt-13b-fcfs-context-fcfs-decode-batch1-1111/distserve-50-5-overlap.png')
    plt.savefig(save_path)
    # plt.show()

# batch size > 1
# path = "./results/opt-13b-fcfs-context-fcfs-decode-500-5-1111/distserve-500-5.exp"
# path = "./results/opt-13b-priority-bounded-sjf-context-priority-bounded-sjf-decode-500-5-1111/distserve-500-5.exp"
# path = "./results/opt-13b-priority-sjf-context-priority-sjf-decode-500-5-1111/distserve-500-5.exp"

# batch size = 1
# path = './results/opt-13b-fcfs-context-fcfs-decode-batch1-1111/distserve-50-5.exp'
# path = './results/opt-13b-priority-sjf-context-priority-sjf-decode-batch1-1111/distserve-50-5.exp'

paths = [
        "./results/opt-13b-bounded-sjf-context-sjf-decode-1111/distserve-50-5.exp",
         "./results/opt-13b-bounded-sjf-context-bounded-sjf-decode-1111/distserve-50-5.exp",
         "./results/opt-13b-bounded-sjf-context-fcfs-decode-1111/distserve-50-5.exp",
         "./results/opt-13b-fcfs-context-priority-bounded-sjf-decode-1111/distserve-50-5.exp",
         "./results/opt-13b-fcfs-context-fcfs-decode-1111/distserve-50-5.exp",
         "./results/opt-13b-fcfs-context-priority-sjf-decode-1111/distserve-50-5.exp",
         "./results/opt-13b-priority-sjf-context-fcfs-decode-1111/distserve-50-5.exp",
         "./results/opt-13b-priority-sjf-context-priority-bounded-sjf-decode-1111/distserve-50-5.exp",
         "./results/opt-13b-priority-sjf-context-priority-sjf-decode-1111/distserve-50-5.exp"
         ]

# with open(path, 'r') as file:
#     data = json.load(file)

# plot_request_intervals(data, path)

for path in paths:
    with open(path, 'r') as file:
        data = json.load(file)

    plot_request_intervals(data, path)
