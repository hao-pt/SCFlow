import torch
import subprocess
import time
import logging
import argparse

# Takes about 8GB
ndim = 25_000
logging.basicConfig(format='[%(asctime)s] %(filename)s [%(levelname).1s] %(message)s', level=logging.DEBUG)
parser = argparse.ArgumentParser('Submit job when gpu is available')
parser.add_argument('--cmd', type=str, default="ls",
                    help='command line')
args = parser.parse_args()

def get_gpu_usage():
    command = "nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits"
    result = subprocess.run(command.split(), capture_output=True, text=True)
    info = result.stdout.strip().split(",")
    mem_total, mem_used, mem_free = map(lambda x: int(x), [info[0], info[1], info[-1]])
    logging.info(f"GPU Stats: Total: {mem_total}, Free: {mem_free} Used: {mem_used}")
    return mem_used / mem_free


def run_job(cmd):
    logging.info(f"Run {cmd}")
    subprocess.run(cmd.split())
    

def main():
    while True:
        usage = get_gpu_usage()
        if usage < 0.05:
            logging.debug("Running GPU job")
            run_job(args.cmd)
            break
        else:
            logging.debug("Waiting for 60 seconds")
            time.sleep(60)


if __name__ == "__main__":
    main()
