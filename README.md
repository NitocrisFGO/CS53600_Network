# CS536 - Data Communication and Computer Networks

This repository contains assignments and projects for CS536 course at Purdue University.

## Installation

This repository is tested with Python 3.12. The easiest way to install this repository is using:

```bash
# Using uv (recommended)
uv pip install -e .

# Or using pip
pip install -e .
```

Alternatively, to set up a development environment with uv:

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

## Assignment 1: Network Latencies, Ping and Traceroute

A Python tool that measures network latency to various servers worldwide, geolocates them, and visualizes the relationship between geographic distance and round-trip time (RTT).

### Features

- Fetches IP addresses from iperf3serverlist.net API or from a custom file
- Pings each host and measures latency statistics (min/avg/max RTT)
- Geolocates IP addresses and calculates geographic distance from your location
- Generates scatter plots showing Distance vs RTT with trend lines
- Handles non-responsive servers gracefully
- Fully automated with customizable parameters
- Outputs results as PDF plots

### Side Notes
- Part 2 will randomly select 5 IP addresses from https://iperf3serverlist.net/api/servers

### Usage

Run both part 1 and part 2:
```
python -m cs536.assignment_1.run_assignment_1
```

Run the ping script as a module:

```bash
# Basic usage - fetch IPs from API and process all
python -m cs536.assignment_1.ping

# Limit number of IPs to process
python -m cs536.assignment_1.ping --limit 10

# Use a custom IP list file
python -m cs536.assignment_1.ping --file ips.txt --output my_plot.pdf

# Customize ping parameters
python -m cs536.assignment_1.ping --count 10 --timeout 5 --delay 0.5

# Show help
python -m cs536.assignment_1.ping --help
```

### Command-Line Arguments

- `--file`: Path to file containing IP addresses (one per line). If not provided, fetches from API.
- `--output`: Output PDF filename for the plot (default: `distance_vs_rtt.pdf`)
- `--count`: Number of ping packets to send (default: 4)
- `--timeout`: Ping timeout in seconds (default: 3)
- `--limit`: Maximum number of IPs to process (default: no limit)
- `--delay`: Delay between requests in seconds for rate limiting (default: 1.0)


### Output

The script generates:
- A PDF scatter plot showing the relationship between geographic distance and RTT
- Console logs with detailed progress information
- Summary statistics (successful/failed pings and geolocations)

### Example

```bash
# Process 20 IPs from the API
source .venv/bin/activate
python -m cs536.assignment_1.ping --limit 20 --output assignment1_results.pdf
```

### Dependencies

- `requests` - HTTP requests for API calls
- `matplotlib` - Plotting library
- `numpy` - Numerical computations for trend lines
- `loguru` - Enhanced logging
- `tyro` - CLI argument parsing

## Development

### Code Formatting

This project uses Ruff for code formatting and linting:

```bash
# Format code
ruff format

# Run linter
ruff check

# Auto-fix linting issues
ruff check --fix
```
