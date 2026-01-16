
# LogAn (Log Analyzer)

**LogAn** is an intelligent log analysis tool that extracts key insights for SREs/Support Engineers/Developers, to identify and diagnose ongoing issues from logs. 
It generates two reports: 
(1) Summary Report presents a table of the representative log lines — each with its predicted golden signals and fault categories — along with the frequency of its occurrence. By using this approach, we've found that the tool can reduce the data volume by up to 90%, since most log lines are informational. (2) Diagnosis Report presents a chronologically ordered set of relevant log windows with user-configurable granularity (e.g., 30s, 1m).


![Architecture](./docs/asset/Logan%20architecture.png)


### Key Features

![Features](./docs/asset/Logan%20features.png)

<!-- 
- **🔍 Intelligent Log Parsing**: Utilizes Drain3 algorithm to automatically extract log templates and identify patterns
- **⚠️ Anomaly Detection**: Identifies unusual log patterns and potential issues using machine learning models
- **📊 Interactive Reports**: Generates rich HTML reports with visualizations and searchable tables -->

## How to Run

### Option 1 - Using Containers (Recommended)

#### Running Container Image

We provide a container image that you can run standalone or integrate into automated pipelines to analyze logs.

```bash
mkdir -p ./tmp/output

podman run --rm \
    -v ./examples/:/data/input/:z \
    -v ./tmp/output/:/data/output/:z \
    -e LOGAN_INPUT_FILES="/data/input/Linux_2k.log" \
    -e LOGAN_OUTPUT_DIR=/data/output/ \
    ghcr.io/Log-Analyzer/LogAn:latest
```


#### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOGAN_INPUT_FILES` | `/data/input` | Input files/directories (comma-separated for multiple) |
| `LOGAN_OUTPUT_DIR` | `/data/output` | Output directory for analysis results |
| `LOGAN_TIME_RANGE` | `all-data` | Time range filter: `all-data`, `1-day`, `2-day`, ..., `1-week`, `2-week`, `1-month` |
| `LOGAN_MODEL_TYPE` | `zero_shot` | Model type: `zero_shot`, `similarity`, `custom` |
| `LOGAN_MODEL` | `crossencoder` | Model for classification: `bart`, `crossencoder`, or custom HuggingFace model |
| `LOGAN_DEBUG_MODE` | `true` | Enable debug mode (saves additional metadata files) |
| `LOGAN_PROCESS_LOG_FILES` | `true` | Process `.log` files found in directories |
| `LOGAN_PROCESS_TXT_FILES` | `false` | Process `.txt` files found in directories |
| `LOGAN_CLEAN_UP` | `false` | Clean output directory before running |


#### Build Container Image

If you want to customize the container image, you can clone this repository and build it manually:

```bash
podman build -t logan -f Containerfile .
```

### Option 2 - Local

You can also directly clone this repository and set up the project without using a container. Make sure you have Python 3 and uv available on your system.

```bash
# Setup venv
uv venv
source .venv/bin/activate

uv pip install -r requirements.txt

# Run Log Analysis
uv run logan analyze \
    -f "examples/Linux_2k.log" \
    -o "tmp/output"
```


## How to View the Reports (Output)
```bash
uv run logan view -d "tmp/output"

# server should be available at http://localhost:8000/log_diagnosis
``` 

## Examples

Check out [tutorials](./examples/tutorials/) for more examples on advanced usages.


## 🔥 Citation 

If you use LogAn for publication, please cite the following research papers: 

- Pranjal Gupta, Karan Bhukar, Harshit Kumar, Seema Nagar, Prateeti Mohapatra, and Debanjana Kar. 2025. [**Scalable and Efficient Large-Scale Log Analysis with LLMs: An IT Software Support Case Study**](https://arxiv.org/abs/2511.14803). AAAI, 2026.

- Pranjal Gupta, Karan Bhukar, Harshit Kumar, Seema Nagar, Prateeti Mohapatra, and Debanjana Kar. 2025. [**LogAn: An LLM-Based Log Analytics Tool with Causal Inferencing**](https://doi.org/10.1145/3680256.3721246). ICPE, 2025.

- Pranjal Gupta, Harshit Kumar, Debanjana Kar, Karan Bhukar, Pooja Aggarwal and Prateeti Mohapatra.2023 [**Learning Representations on Logs for AIOps**](https://arxiv.org/abs/2308.11526). IEEE CLOUD, 2023.

## Authors & Contributors

This project was originally developed by **IBM Research** and is actively supported and maintained by **Red Hat**.

### IBM Research

- Pranjal Gupta
- Harshit Kumar
- Prateeti Mohapatra

### Red Hat

- Pradeep Surisetty
- Pravin Satpute
- Rahul Shetty
- Jan Hutar
- Nikhil Jain


We welcome contributions from the community!

