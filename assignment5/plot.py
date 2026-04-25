import os
import pandas as pd
import matplotlib.pyplot as plt


INPUT_CSV = "results/final_results.csv"
OUTPUT_DIR = "results/plots"

BROADCAST_ALGOS = [
    "binary_tree_broadcast",
    "binomial_tree_broadcast",
]

ALLGATHER_ALGOS = [
    "ring_allgather",
    "recursive_doubling_allgather",
    "swing_allgather",
]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def bytes_to_label(x: int) -> str:
    if x >= 1024 * 1024:
        return f"{x // (1024 * 1024)}MB"
    if x >= 1024:
        return f"{x // 1024}KB"
    return f"{x}B"


def validate_columns(df: pd.DataFrame):
    required = {"algorithm", "world_size", "msg_bytes", "time_ms"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")


def validate_algorithms(df: pd.DataFrame):
    algos = set(df["algorithm"].unique())
    required = set(BROADCAST_ALGOS + ALLGATHER_ALGOS)
    missing = required - algos
    if missing:
        raise ValueError(f"CSV is missing required algorithms: {sorted(missing)}")


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    # 对重复实验按 median 聚合，更稳
    grouped = (
        df.groupby(["algorithm", "world_size", "msg_bytes"], as_index=False)["time_ms"]
        .median()
        .sort_values(["algorithm", "world_size", "msg_bytes"])
    )
    return grouped


def validate_plot_coverage(
    df: pd.DataFrame,
    algos: list[str],
    fixed_world_size: int,
    fixed_msg_bytes: int,
    all_msg_sizes: list[int],
    all_world_sizes: list[int],
):
    # 检查 message-size 图所需数据
    for algo in algos:
        sub = df[(df["algorithm"] == algo) & (df["world_size"] == fixed_world_size)]
        msg_set = set(sub["msg_bytes"].tolist())
        missing = set(all_msg_sizes) - msg_set
        if missing:
            raise ValueError(
                f"Missing data for message-size plot: algo={algo}, "
                f"world_size={fixed_world_size}, missing msg_bytes={sorted(missing)}"
            )

    # 检查 world-size 图所需数据
    for algo in algos:
        sub = df[(df["algorithm"] == algo) & (df["msg_bytes"] == fixed_msg_bytes)]
        ws_set = set(sub["world_size"].tolist())
        missing = set(all_world_sizes) - ws_set
        if missing:
            raise ValueError(
                f"Missing data for world-size plot: algo={algo}, "
                f"msg_bytes={fixed_msg_bytes}, missing world_sizes={sorted(missing)}"
            )


def plot_msgsize_vs_time(
    df: pd.DataFrame,
    algos: list[str],
    fixed_world_size: int,
    title: str,
    output_name: str,
):
    plt.figure(figsize=(8, 5))

    msg_sizes = sorted(df["msg_bytes"].unique())

    for algo in algos:
        sub = df[(df["algorithm"] == algo) & (df["world_size"] == fixed_world_size)]
        sub = sub.sort_values("msg_bytes")
        plt.plot(sub["msg_bytes"], sub["time_ms"], marker="o", label=algo)

    plt.xscale("log", base=2)
    plt.xticks(msg_sizes, [bytes_to_label(x) for x in msg_sizes], rotation=30)
    plt.xlabel("Message size")
    plt.ylabel("Completion time (ms)")
    plt.title(f"{title} (world_size={fixed_world_size})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, output_name), dpi=300)
    plt.close()


def plot_worldsize_vs_time(
    df: pd.DataFrame,
    algos: list[str],
    fixed_msg_bytes: int,
    title: str,
    output_name: str,
):
    plt.figure(figsize=(8, 5))

    world_sizes = sorted(df["world_size"].unique())

    for algo in algos:
        sub = df[(df["algorithm"] == algo) & (df["msg_bytes"] == fixed_msg_bytes)]
        sub = sub.sort_values("world_size")
        plt.plot(sub["world_size"], sub["time_ms"], marker="o", label=algo)

    plt.xticks(world_sizes)
    plt.xlabel("Number of processes / ranks")
    plt.ylabel("Completion time (ms)")
    plt.title(f"{title} (message size={bytes_to_label(fixed_msg_bytes)})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, output_name), dpi=300)
    plt.close()


def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    ensure_dir(OUTPUT_DIR)

    df = pd.read_csv(INPUT_CSV)
    validate_columns(df)
    validate_algorithms(df)

    # 聚合重复实验
    df = aggregate_results(df)

    fixed_world_size = 8
    fixed_msg_bytes = 1048576  # 1MB

    all_msg_sizes = sorted(int(x) for x in df["msg_bytes"].unique())
    all_world_sizes = sorted(int(x) for x in df["world_size"].unique())

    # 先做完整性检查，确保画出来的图完全符合题目要求
    validate_plot_coverage(
        df, BROADCAST_ALGOS, fixed_world_size, fixed_msg_bytes, all_msg_sizes, all_world_sizes
    )
    validate_plot_coverage(
        df, ALLGATHER_ALGOS, fixed_world_size, fixed_msg_bytes, all_msg_sizes, all_world_sizes
    )

    # 1) Broadcast: message size vs completion time
    plot_msgsize_vs_time(
        df=df,
        algos=BROADCAST_ALGOS,
        fixed_world_size=fixed_world_size,
        title="Broadcast: Message Size vs Completion Time",
        output_name="broadcast_msgsize_vs_time.png",
    )

    # 2) AllGather: message size vs completion time
    plot_msgsize_vs_time(
        df=df,
        algos=ALLGATHER_ALGOS,
        fixed_world_size=fixed_world_size,
        title="AllGather: Message Size vs Completion Time",
        output_name="allgather_msgsize_vs_time.png",
    )

    # 3) Broadcast: world size vs completion time
    plot_worldsize_vs_time(
        df=df,
        algos=BROADCAST_ALGOS,
        fixed_msg_bytes=fixed_msg_bytes,
        title="Broadcast: Number of Ranks vs Completion Time",
        output_name="broadcast_worldsize_vs_time.png",
    )

    # 4) AllGather: world size vs completion time
    plot_worldsize_vs_time(
        df=df,
        algos=ALLGATHER_ALGOS,
        fixed_msg_bytes=fixed_msg_bytes,
        title="AllGather: Number of Ranks vs Completion Time",
        output_name="allgather_worldsize_vs_time.png",
    )

    print("Plots generated successfully:")
    print(os.path.join(OUTPUT_DIR, "broadcast_msgsize_vs_time.png"))
    print(os.path.join(OUTPUT_DIR, "allgather_msgsize_vs_time.png"))
    print(os.path.join(OUTPUT_DIR, "broadcast_worldsize_vs_time.png"))
    print(os.path.join(OUTPUT_DIR, "allgather_worldsize_vs_time.png"))


if __name__ == "__main__":
    main()