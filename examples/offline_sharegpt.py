import argparse
import dataclasses
from typing import List
import marshal
from distserve import OfflineLLM, SamplingParams
from distserve.config import (
    ModelConfig,
    DisaggParallelConfig,
    ParallelConfig,
    CacheConfig,
    ContextStageSchedConfig,
    DecodingStageSchedConfig
)

@dataclasses.dataclass
class TestRequest:
    """
    TestRequest: A request for testing the server's performance
    """
    
    prompt: str
    prompt_len: int
    output_len: int
    
@dataclasses.dataclass
class Dataset:
    """
    Dataset: A dataset for testing the server's performance
    """
 
    dataset_name: str	# "sharegpt" / "alpaca" / ...
    reqs: List[TestRequest]
    
    def dump(self, output_path: str):
        marshal.dump({
            "dataset_name": self.dataset_name,
            "reqs": [(req.prompt, req.prompt_len, req.output_len) for req in self.reqs]
        }, open(output_path, "wb"))
    
    @staticmethod
    def load(input_path: str):
        loaded_data = marshal.load(open(input_path, "rb"))
        return Dataset(
            loaded_data["dataset_name"],
            [TestRequest(req[0], req[1], req[2]) for req in loaded_data["reqs"]]
        )

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='The model to use', default='meta-llama/Llama-2-7b-hf')
args = parser.parse_args()

# load sharegpt dataset
dataset = Dataset.load('../evaluation/docs/datasets/sharegpt.ds')
prompts = [req.prompt for req in dataset.reqs[:5]]


# Create a sampling params object.
sampling_params = SamplingParams(
    temperature=0.8, top_p=0.95, max_tokens=64, stop=["\n"]
)

# Create an LLM for offline inference.
llm = OfflineLLM(
    model_config=ModelConfig(
        model=args.model,
        tokenizer=None
    ),
    disagg_parallel_config=DisaggParallelConfig(
        context=ParallelConfig(
            tensor_parallel_size=1,
            pipeline_parallel_size=1
        ),
        decoding=ParallelConfig(
            tensor_parallel_size=1,
            pipeline_parallel_size=1
        )
    ),
    cache_config=CacheConfig(
        block_size=16,
        max_num_blocks_per_req=1024,
        gpu_memory_utilization=0.9,
        cpu_swap_space=1.0
    ),
    context_sched_config=ContextStageSchedConfig(
        policy="fcfs",
        max_batch_size=4,
        max_tokens_per_batch=16384
    ),
    decoding_sched_config=DecodingStageSchedConfig(
        policy="fcfs",
        max_batch_size=4,
        max_tokens_per_batch=16384
    )
)

# Generate texts from the prompts. The output is a list of Request objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)

overall_result = []

i = 0
# Print the outputs.
for prompt, step_outputs in zip(prompts, outputs):
    # new_token_ids = [step_output.new_token_id for step_output in step_outputs]
    # output_text = llm.tokenizer.decode(new_token_ids)
    print(
        f"Prompt: {prompt!r}, Generated text: {' '.join([step_output.new_token for step_output in step_outputs])} ({len(step_outputs)} tokens generated)."
    )
    result = dict()
    result["prompt"] = prompt
    result["prompt_id"] = i
    result["output_len"] = len(step_outputs)
    result["output"] = [step_output.new_token for step_output in step_outputs]
    overall_result.append(result)
    i += 1

import pickle
import json

# pickle.dump(overall_result, open("offline_sharegpt_result_100_prompts.pkl", "wb"))
json.dump(overall_result, open("offline_sharegpt_result_100_prompts_llama.json", "w"))
