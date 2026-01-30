# CS565 – Assignment 1: RTT Measurement and Traceroute Analysis

This repository contains the implementation for **Assignment 1** of the CS565 course.  
The project focuses on measuring network latency using **ping** and **traceroute**, analyzing the relationship between **RTT, hop count, and geographic distance**, and visualizing the results.

---

## Overview

The script performs the following tasks:

### Part 1: Ping and Geographic Distance

- Measures **min / avg / max RTT** to a set of IP addresses using `ping`
- Automatically retrieves and includes **the local machine’s public IP address**
- Geolocates each destination IP using publicly available data
- Computes **geographic distance** between the local machine and each destination
- Generates a **scatter plot of Distance vs RTT**
- Saves raw measurement results in JSON format

### Part 2: Traceroute and Hop Analysis

- Randomly selects **5 destination IP addresses**
- Runs `traceroute` (`tracert` on Windows) to collect hop-by-hop RTTs
- Computes per-hop latency increases (approximate)
- Generates:
  - A **stacked bar chart** showing latency breakdown per hop
  - A **scatter plot of Hop Count vs Total RTT**

---

## Requirements

- **Python 3.10+**
- Tested on **Windows** (uses `ping -n` and `tracert`)
- Internet connection (required for ping and geolocation)

### Python Dependencies

- `requests`
- `matplotlib`

Install dependencies with:

```bash
pip install requests matplotlib
