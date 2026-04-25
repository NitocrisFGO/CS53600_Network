import os
import subprocess

OUTPUT = "results/final_results.csv"

# 先删掉旧结果，避免混在一起
if os.path.exists(OUTPUT):
    os.remove(OUTPUT)

message_sizes = [
    1024,        # 1KB
    4096,        # 4KB
    16384,       # 16KB
    65536,       # 64KB
    262144,      # 256KB
    1048576,     # 1MB
    4194304,     # 4MB
    16777216,    # 16MB
    33554432,    # 32MB
]
world_sizes = [2, 4, 8]

broadcast_algos = ["binary", "binomial"]
allgather_algos = ["ring", "rd", "swing"]

# 每个配置跑几次。先用 3 次，后面想更稳可以改成 5
repeats = 3

def run_cmd(cmd):
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)

# 1) Broadcast: fixed world_size=8, vary message size
for algo in broadcast_algos:
    for msg_bytes in message_sizes:
        for _ in range(repeats):
            run_cmd([
                "python", "benchmark.py",
                "--world-size", "8",
                "--msg-bytes", str(msg_bytes),
                "--collective", "broadcast",
                "--algo", algo,
                "--output", OUTPUT,
            ])

# 2) AllGather: fixed world_size=8, vary message size
for algo in allgather_algos:
    for msg_bytes in message_sizes:
        for _ in range(repeats):
            run_cmd([
                "python", "benchmark.py",
                "--world-size", "8",
                "--msg-bytes", str(msg_bytes),
                "--collective", "allgather",
                "--algo", algo,
                "--output", OUTPUT,
            ])

# 3) Broadcast: fixed msg_bytes=1MB, vary world size
for algo in broadcast_algos:
    for ws in world_sizes:
        if ws == 8:
            continue
        for _ in range(repeats):
            run_cmd([
                "python", "benchmark.py",
                "--world-size", str(ws),
                "--msg-bytes", "1048576",
                "--collective", "broadcast",
                "--algo", algo,
                "--output", OUTPUT,
            ])

# 4) AllGather: fixed msg_bytes=1MB, vary world size
for algo in allgather_algos:
    for ws in world_sizes:
        if ws == 8:
            continue
        for _ in range(repeats):
            run_cmd([
                "python", "benchmark.py",
                "--world-size", str(ws),
                "--msg-bytes", "1048576",
                "--collective", "allgather",
                "--algo", algo,
                "--output", OUTPUT,
            ])

print(f"\nAll experiments finished. Results saved to {OUTPUT}")