"""
Start a proxy and (potential a lot of, if data parallel is enabled) API servers.
The proxy acts as a load balancer which uses round-robin to distribute the requests to the API servers.
"""
import os, sys
import subprocess
import argparse
import multiprocessing

from backends import BACKEND_TO_PORTS

ORACLE_PATH = "2-benchmark-serving/oracle.pkl"

MODEL_TO_PARALLEL_PARAMS = {
    "facebook/opt-125m": {
        "vllm": 1,
        "deepspeed": 1,
        "distserve": (1, 1, 1, 1)
    },
    "facebook/opt-1.3b": {
        "vllm": 1,
        "deepspeed": 1,
        "distserve": (1, 1, 1, 1)
    },
    "facebook/opt-6.7b": {
        "vllm": 1,
        "deepspeed": 1,
        "distserve": (1, 1, 1, 1)   # (context_tp, context_pp, decoding_tp, decoding_pp)
    },
    "facebook/opt-13b": {
        "vllm": 1,
        "deepspeed": 1,
        "distserve": (1, 1, 1, 1)   # TODO adjust me
    },
    "facebook/opt-66b": {
        "vllm": 4,
        "deepspeed": 4,
        "distserve": (4, 1, 2, 2)
    },
    "facebook/opt-175b": {
        "vllm": 8,
        "deepspeed": 8,
        "distserve": (3, 3, 4, 3)
    },
}

def api_server_starter_routine(
    port: int,
    args: argparse.Namespace
):
    """
    Start the target API server on the target port
    """
    use_dummy_weight = os.environ.get("USE_DUMMY_WEIGHT", "0") in ["1", "true", "True"]
    if args.backend == "vllm":
        tp_world_size = MODEL_TO_PARALLEL_PARAMS[args.model]["vllm"]
        script = f"""
conda activate vllm;
python -u -m vllm.entrypoints.api_server \\
    --host 0.0.0.0 --port {port} \\
    --engine-use-ray --worker-use-ray --disable-log-requests \\
    --model {args.model} --dtype half \\
    {"--load-format dummy" if use_dummy_weight else ""} \\
    -tp {tp_world_size} \\
    --block-size 16 --seed 0 \\
    --swap-space 16 \\
    --gpu-memory-utilization 0.95 \\
    --max-num-batched-tokens 16384 \\
    --max-num-seqs 1024 \\
        """

    elif args.backend == "deepspeed":
        tp_world_size = MODEL_TO_PARALLEL_PARAMS[args.model]["deepspeed"]
        if use_dummy_weight:
            print("WARNING: DeepSpeed does not support dummy weights, will use real weights.")
        script = f"""
conda activate deepspeed-mii;
set -u https_proxy; set -u http_proxy; set -u all_proxy;
set -u HTTPS_PROXY; set -u HTTP_PROXY; set -u ALL_PROXY;
python -m mii.entrypoints.api_server \\
    --model {args.model} \\
    --port {port} \\
    --host 0.0.0.0 \\
    --max-length 2048 \\
    --tensor-parallel {tp_world_size}
"""
    

    elif args.backend == "distserve":
        context_tp, context_pp, decoding_tp, decoding_pp = MODEL_TO_PARALLEL_PARAMS[args.model]["distserve"]
        script = f"""
# conda activate distserve;
conda run -n distserve --live-stream python -m distserve.api_server.distserve_api_server \\
    --host 0.0.0.0 \\
    --port {port} \\
    --model {args.model} \\
    --tokenizer {args.model} \\
    {"--use-dummy-weights" if use_dummy_weight else ""} \\
    \\
    --context-tensor-parallel-size {context_tp} \\
    --context-pipeline-parallel-size {context_pp} \\
    --decoding-tensor-parallel-size {decoding_tp} \\
    --decoding-pipeline-parallel-size {decoding_pp} \\
    \\
    --block-size 16 \\
    --max-num-blocks-per-req 128 \\
    --gpu-memory-utilization 0.95 \\
    --swap-space 16 \\
    \\
    --context-sched-policy {args.context_sched_policy} \\
    --context-max-batch-size 128 \\
    --context-max-tokens-per-batch 8192 \\
    \\
    --decoding-sched-policy {args.decoding_sched_policy} \\
    --decoding-max-batch-size 1024 \\
    --decoding-max-tokens-per-batch 65536 \\
    \\
    --priority-sjf-oracle-path {ORACLE_PATH} \\
"""
    
    print(f"Starting server with command {script}")
    subprocess.call(["zsh", "-c", script])


def metadata_server_process(port, args: argparse.Namespace):
    """
    Start a small HTTP server, which returns the metadata of the API servers
    as JSON
    """
    import json
    import http.server
    import socketserver
    
    class MetadataServer(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(args.__dict__).encode())
        def log_message(self, format, *args):
            pass
    
    with socketserver.TCPServer(("", port), MetadataServer) as httpd:
        print("The metadata server is serving at port", port)
        httpd.serve_forever()
    
    
def main(args: argparse.Namespace):
    print(args)
    port = BACKEND_TO_PORTS[args.backend]
    process = multiprocessing.Process(
        target=metadata_server_process,
        args=(port+1, args,)
    )
    process.start()
    api_server_starter_routine(
        port,
        args
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        help="The serving backend"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The model to be served"
    )
    parser.add_argument(
        "--context-sched-policy",
        choices=["fcfs", "priority-sjf", "priority-bounded-sjf"],
        default="fcfs",
    )
    parser.add_argument(
        "--decoding-sched-policy",
        choices=["fcfs", "priority-sjf", "priority-bounded-sjf"],
        default="fcfs",
    )
    args = parser.parse_args()
    main(args)
    
