import math
import random
import subprocess
import re
import matplotlib.pyplot as plt
import json
import platform

import requests


def traceroute(ip):
    cmd = ["tracert", ip]
    result = subprocess.run(cmd, capture_output=True, text=True)
    hops = []

    for line in result.stdout.splitlines():
        # get average RTT（ms）for every router
        # filter: just get lines have value, throw lines like (* * * time out)
        times = re.findall(r"(\d+)\s*ms", line)
        if times:
            avg_rtt = round(sum(map(float, times)) / len(times), 2)
            hops.append(avg_rtt)

    return hops

def latency_breakdown(ips):
    hop_rtts_list = []
    hop_rtt_increase = []

    # get hop list for every ip address.
    for ip in ips:
        hops = traceroute(ip)
        hop_rtts_list.append(hops)
        rtt_increase = [hops[0]]
        for i in range(len(hops)):
            if i > 0:
                rtt_increase.append(hops[i] - hops[i - 1])
        hop_rtt_increase.append(rtt_increase)

    #############################
    #   GET Stacked bar chart
    #############################
    plt.figure()
    bottom = [0] * len(ips)
    for hop_idx in range(max(len(h) for h in hop_rtt_increase)):
        hop_values = []
        for h in hop_rtt_increase:
            hop_values.append(h[hop_idx] if hop_idx < len(h) else 0)

        plt.bar(ips, hop_values, bottom=bottom)
        bottom = [bottom[i] + hop_values[i] for i in range(len(ips))]

    plt.ylabel("RTT (ms)")
    plt.title("Latency Breakdown per Hop")
    plt.savefig("latency_breakdown.pdf")

    #####################################
    #   GET hop count vs RTT graph
    #####################################
    plt.figure()
    hop_counts = []
    total_rtts = []

    for hop_list in hop_rtts_list:
        hop_counts.append(len(hop_list))
        total_rtts.append(hop_list[-1])

    # 给每个 IP 一个颜色
    colors = plt.cm.tab10(range(len(hop_counts)))

    for i in range(len(hop_counts)):
        plt.scatter(
            hop_counts[i],
            total_rtts[i],
            color=colors[i],
            s=80,
            label=ips[i]
        )

    plt.xlabel("Hop Count")
    plt.ylabel("Total RTT (ms)")
    plt.title("Hop Count vs RTT")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("hop_vs_rtt.pdf")


def ping_host(ip, count=3):

    cmd = ["ping", ip, "-n", str(count)]

    try:
        output = subprocess.check_output(cmd, text=True)
    except subprocess.CalledProcessError:
        return None

    min_r = re.search(r"最短 = (\d+)ms", output)
    max_r = re.search(r"最长 = (\d+)ms", output)
    avg_r = re.search(r"平均 = (\d+)ms", output)

    if not (min_r and avg_r and max_r):
        return None
    return {
            "min": float(min_r.group(1)),
            "max": float(max_r.group(1)),
            "avg": float(avg_r.group(1)),
    }


def geolocate(ip):
    url = f"http://ip-api.com/json/{ip}?fields=status,lat,lon,city,country"
    r = requests.get(url, timeout=5).json()
    if r.get("status") == "success":
        return r
    return None


def get_my_location():
    r = requests.get(
        "http://ip-api.com/json/?fields=status,lat,lon,city,country,query",
        timeout=5,
    ).json()
    if r.get("status") == "success":
        return r
    return None

def get_my_public_ip():
    r = requests.get(
        "http://ip-api.com/json/?fields=status,query",
        timeout=5
    ).json()
    if r.get("status") == "success":
        return r["query"]
    return None

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return round(2 * R * math.asin(math.sqrt(a)), 1)

def ping(ips):
    distances = []
    avg_rtts = []
    data_json = []
    labels = []

    my_location = get_my_location()

    for ip in ips:
        rtt = ping_host(ip)
        ip_address = geolocate(ip)

        if not rtt or not ip_address:
            data_json.append({"ip": ip, "rtt": {
            "min": "None",
            "max": "None",
            "avg": "None",
    }, "coordinates": "None"})
            print(f"  Failed for {ip}\n")
            continue

        distance = haversine(ip_address["lat"], ip_address["lon"], my_location["lat"], my_location["lon"])
        distances.append(distance)
        data_json.append({"ip": ip, "rtt": rtt, "coordinates": [ip_address["lat"], ip_address["lon"]]})
        avg_rtts.append(rtt["avg"])
        labels.append(ip)

    return distances, avg_rtts, labels, data_json


def draw_scatter(distances,avg_rtts):
    # ========================
    #  scatter plot
    # ========================
    plt.figure(figsize=(8, 6))
    plt.scatter(distances, avg_rtts, s=80)

    plt.xlabel("Distance (km)")
    plt.ylabel("Average RTT (ms)")
    plt.title("Distance vs RTT")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("distance_vs_rtt.pdf")



with open("listed_iperf3_servers.json", "r", encoding="utf-8") as f:
    data = json.load(f)

ip_list = []

for ip in data:
    ip_list.append(ip["IP/HOST"])

ip_list.append(get_my_public_ip())

distances, avg_rtts, labels, data_json = ping(ip_list)

draw_scatter(distances,avg_rtts)

with open("ping_result.json", "w", encoding="utf-8") as f:
    json.dump(data_json, f, indent=2, ensure_ascii=False)

ips = random.sample(ip_list, 5)
print(ips)

latency_breakdown(ips)