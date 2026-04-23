import os
import time
import csv
import argparse
import socket
import datetime
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from collectives import (
    binary_tree_broadcast,
    binomial_tree_broadcast,
    ring_allgather,
    recursive_doubling_allgather,
    swing_allgather,
)


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_local_tensor(msg_bytes: int, rank: int):
    # uint8: 1 element = 1 byte
    return torch.full((msg_bytes,), fill_value=rank % 256, dtype=torch.uint8)


def my_broadcast_test(tensor, root=0, algo="binary"):
    if algo == "binary":
        return binary_tree_broadcast(tensor, root=root)
    elif algo == "binomial":
        return binomial_tree_broadcast(tensor, root=root)
    else:
        raise ValueError(f"Unknown broadcast algo: {algo}")


def my_allgather_test(send_tensor, algo="ring"):
    if algo == "ring":
        return ring_allgather(send_tensor)
    elif algo == "rd":
        return recursive_doubling_allgather(send_tensor)
    elif algo == "swing":
        return swing_allgather(send_tensor)
    else:
        raise ValueError(f"Unknown allgather algo: {algo}")


def measure_time(fn, tensor, warmup=3, iters=5):
    for _ in range(warmup):
        dist.barrier()
        fn(tensor)
        dist.barrier()

    times = []
    for _ in range(iters):
        dist.barrier()
        start = time.perf_counter()
        fn(tensor)
        dist.barrier()
        elapsed = time.perf_counter() - start

        elapsed_tensor = torch.tensor([elapsed], dtype=torch.float64)
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
        times.append(elapsed_tensor.item())

    times.sort()
    return times[len(times) // 2]


def save_result(csv_path, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["algorithm", "world_size", "msg_bytes", "time_ms"])
        writer.writerow(row)


def worker(rank, world_size, master_addr, master_port, msg_bytes, output, collective, algo):
    # 显式创建 TCPStore，并强制 use_libuv=False
    store = dist.TCPStore(
        host_name=master_addr,
        port=master_port,
        world_size=world_size,
        is_master=(rank == 0),
        timeout=datetime.timedelta(seconds=30),
        use_libuv=False,
    )

    dist.init_process_group(
        backend="gloo",
        store=store,
        rank=rank,
        world_size=world_size,
    )

    tensor = make_local_tensor(msg_bytes, rank)

    if collective == "broadcast":
        time_s = measure_time(lambda x: my_broadcast_test(x, root=0, algo=algo), tensor)

        expected = torch.zeros(msg_bytes, dtype=torch.uint8)
        ok = torch.equal(tensor, expected)

        algo_name = "binary_tree_broadcast" if algo == "binary" else "binomial_tree_broadcast"

    elif collective == "allgather":
        gathered = None

        def run_allgather(x):
            nonlocal gathered
            gathered = my_allgather_test(x, algo=algo)

        time_s = measure_time(run_allgather, tensor)

        expected = torch.cat([
            torch.full((msg_bytes,), fill_value=r % 256, dtype=torch.uint8)
            for r in range(world_size)
        ])
        ok = torch.equal(gathered, expected)

        if algo == "ring":
            algo_name = "ring_allgather"
        elif algo == "rd":
            algo_name = "recursive_doubling_allgather"
        elif algo == "swing":
            algo_name = "swing_allgather"
        else:
            raise ValueError(f"Unknown allgather algo: {algo}")

    else:
        raise ValueError(f"Unknown collective: {collective}")

    local_ok = torch.tensor([1 if ok else 0], dtype=torch.int32)
    dist.all_reduce(local_ok, op=dist.ReduceOp.MIN)
    global_ok = bool(local_ok.item())

    if rank == 0:
        if not global_ok:
            raise RuntimeError(
                f"Verification failed: algo={algo_name}, world_size={world_size}, msg_bytes={msg_bytes}"
            )
        save_result(output, [algo_name, world_size, msg_bytes, time_s * 1000])
        print(
            f"[OK={global_ok}] algo={algo_name}, world_size={world_size}, msg_bytes={msg_bytes}, time_ms={time_s * 1000:.3f}"
        )

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--msg-bytes", type=int, default=1024)
    parser.add_argument("--output", type=str, default="results/raw_results.csv")
    parser.add_argument("--collective", type=str, default="broadcast",
                        choices=["broadcast", "allgather"])
    parser.add_argument(
        "--algo",
        type=str,
        default="binary",
        choices=["binary", "binomial", "ring", "rd", "swing"],
    )
    args = parser.parse_args()

    master_addr = "127.0.0.1"
    master_port = find_free_port()

    mp.spawn(
        worker,
        args=(
            args.world_size,
            master_addr,
            master_port,
            args.msg_bytes,
            args.output,
            args.collective,
            args.algo,
        ),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    import datetime
    mp.set_start_method("spawn", force=True)
    main()