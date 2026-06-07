
# LogAn (Log Analyzer)

**LogAn** is an intelligent log analysis tool that extracts key insights for SREs/Support Engineers/Developers, to identify and diagnose ongoing issues from logs. 
It generates an interactive **Log Explorer** report that presents unique log templates with their predicted golden signals and fault categories, along with occurrence counts. The explorer supports sidebar filtering, timeline visualization, template drill-down, and optional custom tagging. By clustering log lines into templates, LogAn can reduce the data volume by up to 90%, since most log lines are informational.


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
    ghcr.io/log-analyzer/logan:latest
```


#### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOGAN_INPUT_FILES` | `` | Input files/directories (comma-separated for multiple) |
| `LOGAN_INPUT_GLOB` | `` | Input file pattern (glob) for matching multiple files |
| `LOGAN_OUTPUT_DIR` | `/data/output` | Output directory for analysis results |
| `LOGAN_TIME_RANGE` | `all-data` | Time range filter: `all-data`, `1-day`, `2-day`, ..., `1-week`, `2-week`, `1-month` |
| `LOGAN_MODEL_TYPE` | `zero_shot` | Model type: `zero_shot`, `similarity`, `custom` |
| `LOGAN_MODEL` | `crossencoder` | Model for classification: `bart`, `crossencoder`, or custom HuggingFace model |
| `LOGAN_DEBUG_MODE` | `true` | Enable debug mode (saves additional metadata files) |
| `LOGAN_PROCESS_ALL_FILES` | `false` | Process all text-based files irrespective of file extension |
| `LOGAN_PROCESS_LOG_FILES` | `true` | Process `.log` files found in directories |
| `LOGAN_PROCESS_TXT_FILES` | `false` | Process `.txt` files found in directories |
| `LOGAN_CLEAN_UP` | `false` | Clean output directory before running |
| `LOGAN_TAG_CONFIG` | _(none)_ | Path to custom tag config JSON file. Enables tagging when set |


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


## Custom Tags

LogAn supports custom tagging of log lines using a JSON config file. Each tag rule can use **keywords** (substring match on bracket tokens) and/or **regex patterns** (matched against the full log line). This allows any team to categorize their logs by component, severity, or any custom dimension.

### Usage

```bash
# CLI
logan analyze -f server.log -o ./output --tag-config configs/example_tags.json

# Container
podman run --rm \
    -v ./logs/:/data/input/:z \
    -v ./output/:/data/output/:z \
    -v ./my_tags.json:/config/tags.json:z \
    -e LOGAN_INPUT_FILES="/data/input/" \
    -e LOGAN_OUTPUT_DIR=/data/output/ \
    -e LOGAN_TAG_CONFIG=/config/tags.json \
    ghcr.io/log-analyzer/logan:latest
```

### Config Format

```json
{
  "tags": [
    {
      "name": "Authentication",
      "keywords": ["auth", "login", "pam"],
      "patterns": ["authentication\\s+(failure|success)", "session\\s+(opened|closed)"]
    },
    {
      "name": "Network",
      "keywords": ["socket", "connection"],
      "patterns": ["connection\\s+(refused|reset|timed out)"]
    }
  ],
  "default_tag": "Other"
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `tags` | Yes | Array of tag rules |
| `tags[].name` | Yes | Tag label applied to matching log lines |
| `tags[].keywords` | No | Substring match against bracket tokens extracted from log lines (e.g., `[sshd]`, `process[PID]`) |
| `tags[].patterns` | No | Regex patterns matched against the full log line (case-insensitive) |
| `default_tag` | No | Tag for unmatched lines (default: `"Other"`) |

Each tag is scored by the number of keyword + pattern matches. The tag with the highest score wins. See `configs/example_tags.json` for a complete example.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOGAN_TAG_CONFIG` | _(none)_ | Path to custom tag config JSON file. Enables tagging when set. |


## How to View the Reports (Output)
```bash
uv run logan view -d "tmp/output"

# server should be available at http://localhost:8000/log_diagnosis
``` 

![Log-Explorer](./docs/asset/log-explorer.png)

## MCP Server (for AI Agents)

LogAn exposes its analysis capabilities via the [Model Context Protocol (MCP)](https://modelcontextprotocol.io), allowing AI agents (Claude Desktop, Claude Code, custom agents) to analyze logs programmatically.

### Setup

```bash
# Install with MCP dependencies
uv pip install -e ".[mcp]"
```

### Running the Server

```bash
# stdio transport (for Claude Desktop / Claude Code)
logan-mcp

# HTTP transport (for remote agents)
logan-mcp --transport streamable-http
```

Analysis results are stored in `~/.logan/runs/<timestamp>/` by default.

### Available Tools

| Tool | Description |
|------|-------------|
| `analyze_logs` | Run the full analysis pipeline (preprocessing, Drain3 templatization, anomaly detection). Returns each unique log template with its representative log line, golden signal, fault categories, and occurrence count. |
| `extract_templates` | Extract unique log patterns via Drain3 without ML classification. Fast (seconds) — useful for quickly understanding log structure before deciding to run full analysis. |

### Claude Desktop Configuration

Add to your Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "logan": {
      "command": "logan-mcp"
    }
  }
}
```

### Testing with MCP Inspector

```bash
mcp dev logan/mcp/server.py
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

