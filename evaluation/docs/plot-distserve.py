import os, sys
from matplotlib import pyplot as plt
import json
import numpy as np

plt.style.use('ggplot')

EXP_RESULT_ROOT="results/opt-13b"
SYSNAME="DistServe"
OUTPUT="plots"

if not os.path.exists(OUTPUT):
    os.makedirs(OUTPUT)

rps = [(100, 0.25), (100, 0.5), (100, 1), (100, 1.5), (100, 2), (100, 2.25), (100, 2.5), (100, 3)]
thpt = [(0.27, 249.96, 45.00),(0.52, 487.46, 87.76),(0.98, 921.96, 165.99),(1.37, 1287.47, 231.79),(1.63, 1538.71, 277.03),(1.71, 1613.39, 290.47),(1.79, 1683.81, 303.15),(1.91, 1797.54, 323.63)]

plt.figure()
plt.plot([elem[0] for elem in thpt], [elem[2] for elem in thpt])
plt.xlabel("RPS")
plt.ylabel("Output Tokens Generated / sec")
plt.savefig(f"{OUTPUT}/thpt.png", dpi=400)

avg_tpot = []

for elem in rps:
    if not os.path.exists(f"{OUTPUT}/{elem}"):
        os.makedirs(f"{OUTPUT}/{elem}")

    path = f"{EXP_RESULT_ROOT}/distserve-{elem[0]}-{elem[1]}.exp"
    dataset = json.load(open(f"{path}", "r"))

    output_len = []
    latency = []
    issued = []
    context_begin = []
    prefill_queueing = []
    decode_queueing = []
    tpot = []

    for data in dataset:
        output_len.append(len(data["token_timestamps"]))
        latency.append(data["latency"])
        tpot.append(data["tpot"])

        for event in data["lifecycle_events"]:
            if event["event_type"] == "issued":
                issued.append(event["timestamp"])
            if event["event_type"] == "context_begin":
                context_begin.append(event["timestamp"])

        prefill_queueing.append(context_begin[-1] - issued[-1])

    avg = np.mean(tpot)
    avg_tpot.append(avg)

    output_len = np.array(output_len)
    mask = output_len != 1
    new_arr = output_len[mask]

    plt.figure()
    plt.hist(new_arr, bins=40)
    plt.xlabel("Output Token Length")
    plt.savefig(f"{OUTPUT}/{elem}/output_len_hist.png", dpi=400)

    plt.figure()
    plt.scatter(output_len, latency, s=5)
    plt.xlabel("Output Token Length")
    plt.ylabel("Latency (s)")
    plt.savefig(f"{OUTPUT}/{elem}/latency.png", dpi=400)

    plt.figure()
    plt.scatter(output_len, latency, s=5)
    plt.xlabel("Output Token Length")
    plt.ylabel("Latency (s)")
    plt.savefig(f"{OUTPUT}/{elem}/latency.png", dpi=400)

    plt.figure()
    plt.scatter(output_len, prefill_queueing, s=10)
    plt.xlabel("Output Token Length")
    plt.ylabel("Prefill Queueing Time (s)")
    plt.savefig(f"{OUTPUT}/{elem}/prefill_queuing.png", dpi=400)


plt.figure()
plt.plot([elem[0] for elem in thpt], avg_tpot, marker='.')
plt.xlabel("RPS")
plt.ylabel("Average TPOT (s)")
plt.savefig(f"{OUTPUT}/tpot.png", dpi=400)

# Results:
# rps, tokens/s, output tokens/s
# [(0.27,249.96,45.00),(0.52,487.46,87.76),(0.98,921.96,165.99),(1.37,1287.47,231.79),(1.63,1538.71,277.03),(1.71,1613.39,290.47),(1.79,1683.81,303.15),(1.91,1797.54,323.63)]

