
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

`container.sh` contains wrapper for building and running the LogAn as container.

#### Build Container Image

```
bash container.sh build ## You can change ENV=docker/podman in the file
```

#### Running Container Image

1. Execute `container.sh` as follows: 
    ```bash
    bash container.sh run OUTPUT_DIR LOG_FILE_PATH TIME_RANGE(OPTIONAL) -ProcessLogFiles(OPTIONAL) -ProcessTxtFiles(OPTIONAL) -DebugMode(OPTIONAL) 
    ```
   - OUTPUT_DIR - The directory where the tool's reports are stored
   - LOG_FILE_PATH - Folders/files separated by colon(:). 
   - TIME_RANGE - Run analysis only on log lines that fall in the time range determined by the latest date in the data (Allowed values: [1-6]-day, [1-3]-week, 1-month, all-data)
   - `-ProcessLogFiles` - Enable this to process .LOG files (found in folders). This will not affect .LOG files that you have provided explicitly.
   - `-ProcessTxtFiles` - Enable this to process .TXT files (found in folders). This will not affect .TXT files that you have provided explicitly.
   - `-DebugMode` - Enable this flag to store metadata generated during a job for troubleshooting

   Example:
   ```bash
       bash container.sh run ./tmp/output ./examples/Linux_2k.log all-data
   ```
   In the above example: 
   - `./examples/Linux_2k.log` - File
   - `./tmp/output` - Directory


<!-- 2. After running, you will get the following output:
```bash
   OUTPUT_DIR
   ├── cache
   │   └── version.txt
   ├── data
   │   └── <INPUT_DATA>
   ├── developer_debug_files
   │   ├── default_anomalies.csv
   │   ├── default_anomalies_all_info.csv
   │   ├── default_anomalies_info.csv
   │   ├── default_summ_gs_error.csv
   │   ├── default_summ_gs_info.csv
   │   ├── default_temp_to_rep_log.json
   │   └── default_temp_to_signal_map.json
   ├── inferencing_file
   │   └── inference.csv
   ├── log_diagnosis
   │   ├── default_anomalies.html
   │   ├── default_anomalies_info.html
   │   ├── default_summ_gs_error.html
   │   └── default_summ_gs_info.html
   ├── logs
   │   ├── preprocessed_file_epoch-0.log
   │   ├── preprocessed_file_logs-0.log
   │   ├── preprocessed_file_names-0.log
   │   ├── preprocessed_file_timestamps-0.log
   ├── logs_shards
   │   ├── discarded_logs_with_none_time.log
   │   ├── paths_filenames_test_output.log
   │   └── preprocessed_data.csv
   ├── matcher_output_test
   │   └── default_output_matcher.json
   ├── metrics
   │   ├── preprocessing.json
   │   ├── run.json
   │   └── time.json
   ├── pandaralle_cache
   ├── run
   │   ├── drain_train.log
   │   ├── input_file_generation.log
   │   ├── log_diagnosis.log
   │   ├── preprocess.log
   │   ├── stats.log
   │   └── status.log
   └── test_templates
       └── tm-test.templates.json
```


3. You can open `OUTPUT_DIR/log_diagnosis/default_summ_gs_error.html` and `OUTPUT_DIR/log_diagnosis/default_anomalies.html` to check the log diagnosis output. -->


### Option 2 - Local

```
# Setup venv
uv venv
source .venv/bin/activate

uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements.txt

# Run Log Analysis
export OUTPUT_DIR="./tmp/output"

uv run python run_log_diagnosis.py \
    --input_files "./examples/Linux_2k.log" \
    --output_dir "$OUTPUT_DIR" \
    --model-name "cross-encoder/nli-MiniLM2-L6-H768"
```


## How to View the Reports (Output)
```bash
uv run python -m http.server 8000 --directory "${OUTPUT_DIR}"

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

