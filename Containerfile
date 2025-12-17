FROM registry.access.redhat.com/ubi9/ubi-minimal:9.4-1227.1726694542

# Install necessary packages
RUN microdnf update -y \
    && microdnf install -y tar wget gcc openssl-devel bzip2-devel libffi-devel make zlib-devel \
    && microdnf clean all

RUN microdnf install -y bzip2 gzip xz xz-devel zstd lz4 lz4-devel zip unzip cpio file \
    && microdnf clean all

# Install OpenJDK 11
RUN microdnf install -y java-11-openjdk-devel \
    && microdnf clean all

# Download and install Python 3.10.4
WORKDIR /usr/src
RUN wget https://www.python.org/ftp/python/3.10.4/Python-3.10.4.tgz \
    && tar xzf Python-3.10.4.tgz \
    && cd Python-3.10.4 \
    && ./configure --enable-optimizations \
    && make altinstall

# Install pip for Python 3.10
RUN wget https://bootstrap.pypa.io/get-pip.py \
    && python3.10 get-pip.py \
    && pip3 install --upgrade pip

# Set default Python alternatives
RUN alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.10 1 \
    && alternatives --install /usr/bin/pip3 pip3 /usr/local/bin/pip3.10 1

# Verify installations
RUN java -version && python3 --version && pip3 --version

# Clean up
RUN microdnf clean all && rm -rf /usr/src/Python-3.10.4 /usr/src/Python-3.10.4.tgz /usr/src/get-pip.py

## Getting working directory and env variables
WORKDIR /

# Install uv dependency manager
ADD https://astral.sh/uv/0.9.17/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

## Copying source code
WORKDIR /opt/app

# Install logan package
COPY requirements.txt .
RUN uv venv && uv pip install -r requirements.txt

# Copying source code
COPY . .
RUN uv pip install -e . --no-deps

# Make run.sh executable
RUN chmod +x run.sh

#######################################
# Environment Variable Defaults
#######################################

# Mode: 'analyze' to run log analysis, 'view' to serve report via HTTP
# TODO: Fix view command not working
ENV LOGAN_MODE="analyze"

# Input files/directories (comma-separated for multiple)
# Example: "/data/logs/app.log,/data/logs/system.log" or "/data/logs/"
ENV LOGAN_INPUT_FILES="/data/input"

# Output directory for analysis results
ENV LOGAN_OUTPUT_DIR="/data/output"

# Time range for analysis
# Options: all-data, 1-day, 2-day, ..., 1-week, 2-week, 1-month
ENV LOGAN_TIME_RANGE="all-data"

# Model type for anomaly detection
# Options: zero_shot, similarity, custom
ENV LOGAN_MODEL_TYPE="zero_shot"

# Model to use for classification
# Built-in: bart, crossencoder
# Or specify a custom HuggingFace model name
ENV LOGAN_MODEL="crossencoder"

# Enable debug mode (saves debug files)
ENV LOGAN_DEBUG_MODE="true"

# Process .log files from directories
ENV LOGAN_PROCESS_LOG_FILES="true"

# Process .txt files from directories
ENV LOGAN_PROCESS_TXT_FILES="false"

# Clean up output directory before running
ENV LOGAN_CLEAN_UP="false"

# Port for view mode HTTP server
ENV LOGAN_VIEW_PORT="8000"

# Expose port for view mode
EXPOSE 8000

# Directory to serve in view mode (defaults to LOGAN_OUTPUT_DIR if not set)
ENV LOGAN_VIEW_DIR=""

#######################################

## Running Script
ENTRYPOINT ["./run.sh"]
