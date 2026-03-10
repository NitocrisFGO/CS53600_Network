#!/usr/bin/env python3

import socket
import struct
import time
import csv
import json
import argparse
import random
import ctypes
import os
import select
from pathlib import Path
import matplotlib.pyplot as plt

TCP_INFO = getattr(socket, "TCP_INFO", 11)

PARAM_EXCHANGE = 9
CREATE_STREAMS = 10
TEST_START = 1
TEST_RUNNING = 2
TEST_END = 4
EXCHANGE_RESULTS = 13


class TcpInfo(ctypes.Structure):
    _fields_ = [
        ("tcpi_state", ctypes.c_uint8),
        ("tcpi_ca_state", ctypes.c_uint8),
        ("tcpi_retransmits", ctypes.c_uint8),
        ("tcpi_probes", ctypes.c_uint8),
        ("tcpi_backoff", ctypes.c_uint8),
        ("tcpi_options", ctypes.c_uint8),
        ("tcpi_snd_wscale", ctypes.c_uint8),
        ("tcpi_rcv_wscale", ctypes.c_uint8),

        ("tcpi_rto", ctypes.c_uint32),
        ("tcpi_ato", ctypes.c_uint32),
        ("tcpi_snd_mss", ctypes.c_uint32),
        ("tcpi_rcv_mss", ctypes.c_uint32),

        ("tcpi_unacked", ctypes.c_uint32),
        ("tcpi_sacked", ctypes.c_uint32),
        ("tcpi_lost", ctypes.c_uint32),
        ("tcpi_retrans", ctypes.c_uint32),
        ("tcpi_fackets", ctypes.c_uint32),

        ("tcpi_last_data_sent", ctypes.c_uint32),
        ("tcpi_last_ack_sent", ctypes.c_uint32),
        ("tcpi_last_data_recv", ctypes.c_uint32),
        ("tcpi_last_ack_recv", ctypes.c_uint32),

        ("tcpi_pmtu", ctypes.c_uint32),
        ("tcpi_rcv_ssthresh", ctypes.c_uint32),
        ("tcpi_rtt", ctypes.c_uint32),
        ("tcpi_rttvar", ctypes.c_uint32),
        ("tcpi_snd_ssthresh", ctypes.c_uint32),
        ("tcpi_snd_cwnd", ctypes.c_uint32),
        ("tcpi_advmss", ctypes.c_uint32),
        ("tcpi_reordering", ctypes.c_uint32),

        ("tcpi_rcv_rtt", ctypes.c_uint32),
        ("tcpi_rcv_space", ctypes.c_uint32),

        ("tcpi_total_retrans", ctypes.c_uint32),
    ]


def get_tcp_info(sock):
    buf = sock.getsockopt(socket.IPPROTO_TCP, TCP_INFO, 192)
    return TcpInfo.from_buffer_copy(buf[:ctypes.sizeof(TcpInfo)])


def estimate_bytes_acked(bytes_sent, unacked, mss):
    inflight = unacked * mss
    return max(0, bytes_sent - inflight)


def make_cookie():
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return ''.join(random.choice(alphabet) for _ in range(37)).encode()


def recv_exact(sock: socket.socket, n: int, timeout: float = 1.0) -> bytes:
    """Receive exactly n bytes, but never hang forever."""
    old = sock.gettimeout()
    sock.settimeout(timeout)
    try:
        data = b""
        while len(data) < n:
            chunk = sock.recv(n - len(data))
            if not chunk:
                # peer closed
                raise ConnectionError("socket closed while receiving")
            data += chunk
        return data
    finally:
        sock.settimeout(old)


def recv_json(sock: socket.socket, timeout: float = 1.0):
    length_bytes = recv_exact(sock, 4, timeout=timeout)
    length = struct.unpack("!I", length_bytes)[0]
    raw = recv_exact(sock, length, timeout=timeout)
    return json.loads(raw.decode())


def send_json(sock, obj):
    raw = json.dumps(obj).encode()
    sock.sendall(struct.pack("!I", len(raw)))
    sock.sendall(raw)


def run_test(server, duration, interval, debug=False):
    ctrl = None
    samples = []
    success = False
    data_sock = None
    try:
        ctrl = socket.socket()
        ctrl.settimeout(10.0)
        ctrl.connect((server[0], server[1]))
    except (socket.timeout, OSError) as e:
        if debug:
            print(f"[dbg] ctrl connect failed: {e}")
        if ctrl:
            ctrl.close()
        return samples, False

    cookie = make_cookie()
    ctrl.sendall(cookie)

    test_started = False

    bytes_sent_total = 0

    start_time = None
    next_sample = None

    last_acked_total = 0
    window_bytes = 0

    last_retrans = 0

    chunk = b"\0" * 16384

    while True:

        ctrl.settimeout(10.0)

        try:
            b = ctrl.recv(1)
        except socket.timeout:
            if debug:
                print("[dbg] timeout waiting for ctrl state; abort this server")
            break
        except (ConnectionResetError, BrokenPipeError, OSError) as e:
            if debug:
                print(f"[dbg] ctrl recv failed: {e}; abort this server")
            break

        if not b:
            if debug:
                print("[dbg] ctrl socket closed by server")
            break

        state = b[0]
        print("CTRL state =", state)

        # ----------------------------
        # PARAM_EXCHANGE
        # ----------------------------
        if state == PARAM_EXCHANGE:

            params = {
                "tcp": True,
                "omit": 0,
                "time": duration,
                "num": 1,
                "len": 131072,
                "bandwidth": 0
            }

            send_json(ctrl, params)

        # ----------------------------
        # CREATE_STREAMS
        # ----------------------------
        elif state == CREATE_STREAMS:

            data_sock = socket.socket()
            data_sock.connect((server[0], server[1]))
            data_sock.sendall(cookie)

        # ----------------------------
        # TEST START / RUNNING
        # ----------------------------
        elif state in (TEST_START, TEST_RUNNING):

            test_started = True

            if start_time is None:

                start_time = time.time()
                next_sample = start_time + interval

                info = get_tcp_info(data_sock)

                mss = info.tcpi_snd_mss or 1460
                unacked = info.tcpi_unacked

                last_acked_total = bytes_sent_total - unacked * mss
                last_retrans = info.tcpi_total_retrans

                if debug:
                    print(f"[dbg] start={start_time:.6f} interval={interval}")

            while time.time() - start_time < duration:

                try:
                    sent = data_sock.send(chunk)
                    bytes_sent_total += sent
                except:
                    break

                info = get_tcp_info(data_sock)

                mss = info.tcpi_snd_mss or 1460
                unacked = info.tcpi_unacked

                acked_total = bytes_sent_total - unacked * mss

                delta = acked_total - last_acked_total
                if delta > 0:
                    window_bytes += delta

                last_acked_total = acked_total

                now = time.time()

                if now >= next_sample:

                    goodput_bps = (window_bytes / interval) * 8.0

                    retrans = info.tcpi_total_retrans
                    d_retrans = retrans - last_retrans
                    last_retrans = retrans

                    loss_signal = d_retrans / interval

                    t_sec = next_sample - start_time

                    sample = {
                        "t": round(t_sec, 1),
                        "goodput": goodput_bps,
                        "cwnd": int(info.tcpi_snd_cwnd),
                        "rtt": info.tcpi_rtt / 1000.0,
                        "rttvar": info.tcpi_rttvar / 1000.0,
                        "retrans": int(retrans),
                        "loss": loss_signal,
                    }

                    samples.append(sample)

                    if debug:
                        print("sample", sample)

                    window_bytes = 0
                    next_sample += interval

            ctrl.send(b"\x04")

        # ----------------------------
        # EXCHANGE_RESULTS
        # ----------------------------
        elif state == EXCHANGE_RESULTS:

            ctrl.settimeout(1.0)

            try:
                obj = recv_json(ctrl, timeout=1.0)
                if debug:
                    print("[dbg] got result json")
            except Exception:
                if debug:
                    print("[dbg] no result json (ok)")

            try:
                send_json(ctrl, {})
            except:
                pass

            # 判断是否成功
            if test_started and len(samples) > 0:
                success = True

            break

    ctrl.close()

    if data_sock:
        data_sock.close()

    return samples, success


def write_q1(all_results, outdir):
    """
    all_results: dict[str, list[dict]]
        key   = destination/server name
        value = samples for that destination
                each sample looks like:
                {
                    "t": ...,
                    "goodput": ...
                }
    """
    os.makedirs(outdir, exist_ok=True)

    # -------------------------
    # 1) write all samples
    # -------------------------
    with open(os.path.join(outdir, "goodput_all.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["destination", "time", "goodput_mbps"])

        for dest, samples in all_results.items():
            for s in samples:
                w.writerow([dest, s["t"], s["goodput"] / 1e6])

    # -------------------------
    # 2) per-destination summary
    # -------------------------
    with open(os.path.join(outdir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["destination", "min", "median", "avg", "p95"])

        for dest, samples in all_results.items():
            if not samples:
                continue

            goodputs = sorted([s["goodput"] / 1e6 for s in samples])

            n = len(goodputs)
            min_v = goodputs[0]
            median_v = goodputs[n // 2]
            avg_v = sum(goodputs) / n
            p95_v = goodputs[min(int(n * 0.95), n - 1)]

            w.writerow([dest, min_v, median_v, avg_v, p95_v])

    # -------------------------
    # 3) one plot with all destinations
    # -------------------------
    plt.figure()

    for dest, samples in all_results.items():
        if not samples:
            continue

        t = [s["t"] for s in samples]
        goodputs = [s["goodput"] / 1e6 for s in samples]

        plt.plot(t, goodputs, label=dest)

    plt.xlabel("time (s)")
    plt.ylabel("goodput (Mbps)")
    plt.title("Throughput evolution across all destinations")
    plt.legend(fontsize="x-small")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "goodput_timeseries.pdf"))
    plt.close()

def write_q2(samples, outdir):
    import os, csv
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)

    # ---- 1) write trace.csv ----
    trace_path = os.path.join(outdir, "trace.csv")
    with open(trace_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "goodput_mbps", "cwnd", "rtt_ms", "rttvar_ms", "retrans", "loss"])
        for s in samples:
            w.writerow([
                float(s["t"]),
                float(s["goodput"]) / 1e6,   # Mbps
                int(s["cwnd"]),
                float(s["rtt"]),             # assume ms (matches your prints)
                float(s["rttvar"]),          # assume ms
                int(s["retrans"]),
                float(s["loss"]),
            ])

    # ---- 2) arrays for plotting ----
    t = [float(s["t"]) for s in samples]
    goodput_mbps = [float(s["goodput"]) / 1e6 for s in samples]
    cwnd = [int(s["cwnd"]) for s in samples]
    rtt = [float(s["rtt"]) for s in samples]
    loss = [float(s["loss"]) for s in samples]
    retrans = [int(s["retrans"]) for s in samples]

    # ---- 3) time series plots (recommended) ----

    plt.figure()
    plt.plot(t, cwnd)
    plt.title("cwnd (time series)")
    plt.xlabel("time (s)")
    plt.ylabel("cwnd (segments)")
    plt.savefig(os.path.join(outdir, "cwnd.pdf"))
    plt.close()

    plt.figure()
    plt.plot(t, rtt)
    plt.title("rtt (time series)")
    plt.xlabel("time (s)")
    plt.ylabel("rtt (ms)")
    plt.savefig(os.path.join(outdir, "rtt.pdf"))
    plt.close()

    # loss / retrans time series（loss信号/重传信号常被要求展示）
    plt.figure()
    plt.plot(t, loss)
    plt.title("loss signal (time series)")
    plt.xlabel("time (s)")
    plt.ylabel("loss")
    plt.savefig(os.path.join(outdir, "loss.pdf"))
    plt.close()

    plt.figure()
    plt.plot(t, retrans)
    plt.title("retrans (time series)")
    plt.xlabel("time (s)")
    plt.ylabel("total retrans")
    plt.savefig(os.path.join(outdir, "retrans.pdf"))
    plt.close()

    # ---- 4) scatter plots (the key missing requirement) ----
    # cwnd vs goodput
    plt.figure()
    plt.scatter(cwnd, goodput_mbps, s=12)
    plt.title("cwnd vs goodput")
    plt.xlabel("cwnd (segments)")
    plt.ylabel("goodput (Mbps)")
    plt.savefig(os.path.join(outdir, "scatter_cwnd_goodput.pdf"))
    plt.close()

    # rtt vs goodput
    plt.figure()
    plt.scatter(rtt, goodput_mbps, s=12)
    plt.title("rtt vs goodput")
    plt.xlabel("rtt (ms)")
    plt.ylabel("goodput (Mbps)")
    plt.savefig(os.path.join(outdir, "scatter_rtt_goodput.pdf"))
    plt.close()

    # loss vs goodput
    plt.figure()
    plt.scatter(loss, goodput_mbps, s=12)
    plt.title("loss vs goodput")
    plt.xlabel("loss signal")
    plt.ylabel("goodput (Mbps)")
    plt.savefig(os.path.join(outdir, "scatter_loss_goodput.pdf"))
    plt.close()

def load_servers(path):
    servers = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        host = item["IP/HOST"]
        port_field = item["PORT"]
        if "-" in port_field:
            port = int(port_field.split("-")[0])
        else:
            port = int(port_field)
        servers.append((host, port))
    return servers


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--servers", default="listed_iperf3_servers.json")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--duration", type=int, default=45)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    rng = random.Random(int(time.time()))

    p = Path(args.servers)
    dests = load_servers(p)
    if not dests:
        print(f"[err] dest_file had no usable destinations: {p}")
        return

    print(f"Found {len(dests)} servers")
    successes = []
    used = []
    all_result = {}
    while len(successes) < args.n:
        server = rng.choice(dests)
        key = server[0]
        if key in used:
            continue
        used.append(key)
        print(f"{server}")


        samples, if_sucess = run_test(
            server,
            args.duration,
            args.interval,
            args.debug
        )


        if if_sucess:
            if str(samples[-1]['t']) == f'{args.duration}.0':
                successes.append(server)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                all_result[f'server{len(successes)}'] = samples
                write_q2(samples, f"results/q2/{len(successes)}")
            else:
                if args.debug:
                    print(f'{server[0]} not fit t = 10.')

    write_q1(all_result, f"results/q1")
    print("Done")


if __name__ == "__main__":
    main()