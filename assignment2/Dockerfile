FROM ubuntu:24.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    iperf3 \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

# create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# install python packages into venv
RUN pip install --upgrade pip && \
    pip install numpy pandas matplotlib scikit-learn

CMD ["bash", "run_all.sh"]