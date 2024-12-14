import matplotlib.pyplot as plt
import numpy as np
import json
from structs import Dataset
import pickle
import hashlib

def dump_oracle(requests, dump_path):
    req_mapping = {}
    for req in requests:
        prompt = req.prompt
        prompt_len = req.prompt_len
        output_len = req.output_len

        req_mapping[hashlib.md5(prompt.encode()).hexdigest()] = output_len

    # print(requests[0].prompt)
    with open(dump_path, 'wb') as f:
        pickle.dump(req_mapping, f)


if __name__ == "__main__":
    dataset = Dataset.load('../docs/datasets/sharegpt.ds')
    dump_oracle(dataset.reqs, "oracle.pkl")
