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

## Settting up Python dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN pip3 install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

## Copying source code
WORKDIR /opt/app
COPY . .

## Changing to non-root user
USER 1001

## Running Script
ENTRYPOINT ["./run.sh"]
