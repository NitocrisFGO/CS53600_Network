import math
import torch
import torch.distributed as dist


def binary_tree_broadcast(tensor, root=0):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    vrank = (rank - root + world_size) % world_size

    parent_v = None if vrank == 0 else (vrank - 1) // 2
    left_v = 2 * vrank + 1
    right_v = 2 * vrank + 2

    def to_real(v):
        return (v + root) % world_size

    if parent_v is not None:
        parent = to_real(parent_v)
        dist.recv(tensor, src=parent)

    if left_v < world_size:
        left = to_real(left_v)
        dist.send(tensor, dst=left)

    if right_v < world_size:
        right = to_real(right_v)
        dist.send(tensor, dst=right)

    return tensor


def binomial_tree_broadcast(tensor, root=0):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    vrank = (rank - root + world_size) % world_size

    def to_real(v):
        return (v + root) % world_size

    mask = 1

    while mask < world_size:
        if vrank & mask:
            parent_v = vrank ^ mask
            parent = to_real(parent_v)
            dist.recv(tensor, src=parent)
            break
        mask <<= 1

    mask >>= 1
    while mask > 0:
        child_v = vrank | mask
        if child_v < world_size:
            child = to_real(child_v)
            dist.send(tensor, dst=child)
        mask >>= 1

    return tensor


def ring_allgather(send_tensor):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    chunk_size = send_tensor.numel()
    recv_tensor = torch.empty(world_size * chunk_size, dtype=send_tensor.dtype)

    # 先把自己的块放到属于自己的位置
    recv_tensor[rank * chunk_size : (rank + 1) * chunk_size] = send_tensor

    left = (rank - 1 + world_size) % world_size
    right = (rank + 1) % world_size

    for step in range(world_size - 1):
        send_index = (rank - step) % world_size
        recv_index = (rank - step - 1 + world_size) % world_size

        send_buf = recv_tensor[
            send_index * chunk_size : (send_index + 1) * chunk_size
        ].clone()

        recv_buf = torch.empty(chunk_size, dtype=send_tensor.dtype)

        req_recv = dist.irecv(recv_buf, src=left)
        req_send = dist.isend(send_buf, dst=right)

        req_recv.wait()
        req_send.wait()

        recv_tensor[
            recv_index * chunk_size : (recv_index + 1) * chunk_size
        ] = recv_buf

    return recv_tensor


def recursive_doubling_allgather(send_tensor):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 只支持 world_size 为 2 的幂
    if world_size & (world_size - 1) != 0:
        raise ValueError("recursive_doubling_allgather requires world_size to be a power of 2")

    chunk_size = send_tensor.numel()
    recv_tensor = torch.empty(world_size * chunk_size, dtype=send_tensor.dtype)

    # 自己那一块先放进去
    recv_tensor[rank * chunk_size : (rank + 1) * chunk_size] = send_tensor

    rounds = int(math.log2(world_size))

    for k in range(rounds):
        partner = rank ^ (1 << k)

        block_ranks = 1 << k
        block_elems = block_ranks * chunk_size

        # 我当前拥有的连续块起点
        send_block_start_rank = (rank >> k) << k
        send_block_start = send_block_start_rank * chunk_size

        # 对方当前拥有的连续块起点
        recv_block_start_rank = (partner >> k) << k
        recv_block_start = recv_block_start_rank * chunk_size

        send_buf = recv_tensor[send_block_start : send_block_start + block_elems].clone()
        recv_buf = torch.empty(block_elems, dtype=send_tensor.dtype)

        req_recv = dist.irecv(recv_buf, src=partner)
        req_send = dist.isend(send_buf, dst=partner)

        req_recv.wait()
        req_send.wait()

        recv_tensor[recv_block_start : recv_block_start + block_elems] = recv_buf

    return recv_tensor


def _swing_rho(step: int) -> int:
    # ρ(s) = sum_{i=0}^{s} (-2)^i
    total = 0
    for i in range(step + 1):
        total += (-2) ** i
    return total


def _swing_peer(rank: int, step: int, world_size: int) -> int:
    """
    π(r, s): Swing peer at step s on a 1D torus.
    Even ranks use +ρ(s), odd ranks use -ρ(s), modulo p.
    """
    rho = _swing_rho(step)
    if rank % 2 == 0:
        return (rank + rho) % world_size
    else:
        return (rank - rho) % world_size


def _precompute_swing_allgather_sets(world_size: int):
    """
    Precompute the set of block indices each rank owns before each allgather step.

    For Swing allgather, peers are selected in reverse order compared with
    reduce-scatter, and the amount of data doubles at each step.
    owned_by_round[t][r] = set of block indices rank r owns BEFORE ag step t.
    """
    rounds = int(math.log2(world_size))

    # round 0: each rank only owns its own block
    owned = [{r} for r in range(world_size)]
    owned_by_round = [owned]

    for ag_step in range(rounds):
        rs_step = rounds - 1 - ag_step   # reverse order for allgather
        new_owned = []

        for r in range(world_size):
            peer = _swing_peer(r, rs_step, world_size)
            new_owned.append(owned[r] | owned[peer])

        owned = new_owned
        owned_by_round.append(owned)

    return owned_by_round


def swing_allgather(send_tensor):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 论文里标准版本默认以 power-of-two 为基础；
    # 你的实验如果只跑 2/4/8，这个限制是合理的。
    if world_size & (world_size - 1) != 0:
        raise ValueError("swing_allgather requires world_size to be a power of 2")

    rounds = int(math.log2(world_size))
    chunk_size = send_tensor.numel()

    # 最终输出：按 rank 顺序拼接所有块
    recv_tensor = torch.empty(world_size * chunk_size, dtype=send_tensor.dtype)
    recv_tensor[rank * chunk_size : (rank + 1) * chunk_size] = send_tensor

    # 预计算每一轮前，每个 rank 持有哪些块
    owned_by_round = _precompute_swing_allgather_sets(world_size)

    for ag_step in range(rounds):
        rs_step = rounds - 1 - ag_step   # allgather uses reverse peer order
        peer = _swing_peer(rank, rs_step, world_size)

        send_indices = sorted(owned_by_round[ag_step][rank])
        recv_indices = sorted(owned_by_round[ag_step][peer])

        expected_count = 1 << ag_step
        if len(send_indices) != expected_count or len(recv_indices) != expected_count:
            raise RuntimeError(
                f"Swing AG internal error at rank={rank}, ag_step={ag_step}, "
                f"send_count={len(send_indices)}, recv_count={len(recv_indices)}, "
                f"expected={expected_count}"
            )

        # 按规范：当前步骤发送“自己到目前为止 gathered 的所有块”
        send_buf = torch.cat([
            recv_tensor[i * chunk_size : (i + 1) * chunk_size].clone()
            for i in send_indices
        ])

        recv_buf = torch.empty(expected_count * chunk_size, dtype=send_tensor.dtype)

        req_recv = dist.irecv(recv_buf, src=peer)
        req_send = dist.isend(send_buf, dst=peer)
        req_recv.wait()
        req_send.wait()

        # 按预计算出的块顺序写回
        for j, idx in enumerate(recv_indices):
            src_start = j * chunk_size
            src_end = (j + 1) * chunk_size
            dst_start = idx * chunk_size
            dst_end = (idx + 1) * chunk_size
            recv_tensor[dst_start:dst_end] = recv_buf[src_start:src_end]

    return recv_tensor