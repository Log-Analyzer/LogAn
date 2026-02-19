# Stage 0: grab uv binary from official uv image
FROM ghcr.io/astral-sh/uv:latest AS uvbin

# Final image: start from Red Hat UBI python image
# More details: https://catalog.redhat.com/en/software/containers/ubi9/python-311/63f764b03f0b02a2e2d63fff#overview
FROM registry.access.redhat.com/ubi9/python-311:9.7

USER root

# Install runtime packages
RUN yum update -y \
 && yum install -y \
      java-11-openjdk-headless \
      wget \
      make gcc \
      unzip \
      python3-devel

# Copy uv binary (and helper uvx if present) from upstream image
COPY --from=uvbin /uv /usr/local/bin/uv
COPY --from=uvbin /uvx /usr/local/bin/uvx
RUN chmod 755 /usr/local/bin/uv /usr/local/bin/uvx || true

# Optional: verify
RUN java -version && python3 --version && pip3 --version && uv --version

# ----------------------------
# Switch to UBI default app dir
# ----------------------------
WORKDIR /opt/app-root/src

# Install all Python deps in one layer, then remove build-only packages
COPY --chmod=755 requirements.txt .
RUN uv venv --python python3.11 \
 && uv pip install --no-cache-dir -r requirements.txt \
 && uv pip install --no-cache-dir \
      torch==2.2.2+cpu \
      --index-url https://download.pytorch.org/whl/cpu

# Install logan package
COPY --chmod=755 . .
RUN uv pip install --no-cache-dir . --no-deps

# Clean up build-only packages
RUN yum remove -y make gcc python3-devel \
 && yum clean all \
 && rm -rf /var/cache/yum

# Redirect runtime caches to /tmp so non-root user can write to them
ENV HF_HOME="/tmp/hf_cache"

# Drop to non-root user for runtime
USER 1001

#######################################
# Environment Variable Defaults
#######################################

# Mode: 'analyze' to run log analysis, 'view' to serve report via HTTP
# TODO: Fix view command not working
ENV LOGAN_MODE="analyze"

# Input files/directories (comma-separated for multiple)
# Example: "/data/logs/app.log,/data/logs/system.log" or "/data/logs/"
ENV LOGAN_INPUT_FILES=""
ENV LOGAN_INPUT_GLOB=""

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

# Process all text based files irrespective of the file extension
ENV LOGAN_PROCESS_ALL_FILES="false"

# Process only .log files from directories
ENV LOGAN_PROCESS_LOG_FILES="true"

# Process only .txt files from directories
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
