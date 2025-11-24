#!/bin/bash
# This script runs a series of Python scripts to perform log preprocessing,
# templatization of preprocessed logs, and followed by anomaly and summary report generation in a pipeline. 
# It uses environment variables and command line arguments to manage inputs, outputs, and processing stages.

set -uo pipefail  # Exit on error, unset variables are errors, and propagate errors in pipelines.

# Parse EXTRA_ARGS environment variable into an array.
IFS=' ' read -ra array <<< "$EXTRA_ARGS"
TIME_RANGE=${array[1]}    # Extract the time range from the parsed input.

OPTIONS=("${array[@]:2}")
## ProcessLogFiles is by default selected (set to be true) in the UI and it should be communicated to PROCESS_LOG_FILES var here.
PROCESS_LOG_FILES=false
PROCESS_TXT_FILES=false
DEBUG_MODE=false

for variable in "${OPTIONS[@]}"; do
    case $variable in
        -ProcessLogFiles)
            PROCESS_LOG_FILES=true
            ;;
        -ProcessTxtFiles)
            PROCESS_TXT_FILES=true
            ;;
        -DebugMode)
            DEBUG_MODE=true
            ;;
        *)
    esac
done

# Get the input file/folder paths(concatenated with :) on which analysis needs to be performed from the FILE environment variable.
IFS=':' read -ra INPUT_FILES <<< "$FILE"  # Split input file list into an array.


export ENVIRONMENT="${OPTIONS[-1]}"
# Determine the output directory based on the environment setting.
if [ "$ENVIRONMENT" = "LOCAL" ]; then
    echo "Running the tool locally"
    OUTPUT_DIR="${OUT_DIRECTORY}"  # Use specified output directory for LOCAL.
else
    OUTPUT_DIR=$(dirname "$OUT_DIRECTORY")  # Use the parent directory for non-LOCAL environments.
fi

# Function to send statistics after pipeline stages.
# Example: send_stats true
function send_stats() {
    success=$1  # Capture the success flag.
    echo "{\"success\": $success}" > "${OUTPUT_DIR}/metrics/run.json" 2>&1
}

# Function to handle the exit status of the last command.
# Based on the exit code, it either continues or exits the script with telemetry.
# Example: handle_last_cmd_exit
function handle_last_cmd_exit() {
    last_exit_code=$1 # Capture the exit code of the last command.
   
    if [ "$last_exit_code" -eq "0" ]; then
        # The last step ran successfully. No need to break the pipeline
        return 0
    elif [ "$last_exit_code" -eq "102" ]; then
        # Specific exit code 102 handling for two cases.
        # 1st user has selected None as product name or,
        # 2nd user has provided archive file as input file.
        # In both cases empty summary file is created with appropriate error messages
        echo "Exit Code: $last_exit_code" >> "${OUTPUT_DIR}/run/status.log" 2>&1
        echo "------------- FAILED: $PIPELINE_STAGE -------------" >> "${OUTPUT_DIR}/run/status.log" 2>&1
        exit 1
    else 
        # General failure handling while executing pipeline stages:
        # It logs and send stats to the dashboard and generate a empty summary and anomaly file
        echo "Exit Code: $last_exit_code" >> "${OUTPUT_DIR}/run/status.log" 2>&1
        echo "------------- FAILED: $PIPELINE_STAGE -------------" >> "${OUTPUT_DIR}/run/status.log" 2>&1
        cd ./log_diagnosis && python3 utils.py --output_dir "${OUTPUT_DIR}/log_diagnosis"
        send_stats false
        exit $last_exit_code  # Exit the script with the captured exit code.
    fi
}

# Function to run a specific pipeline stage and handle errors.
# It logs the start, measures execution time, and calls handle_last_cmd_exit for error handling.
function run_pipeline_stage() {
    echo -e "\n------------- STARTING: $PIPELINE_STAGE -------------" >> "${OUTPUT_DIR}/run/status.log" 2>&1

    "$@"  # Execute the command passed to the function.
    status_returned=$?  # Capture the status returned by handle_last_cmd_exit.
    handle_last_cmd_exit $status_returned # Check the command's exit status and handle errors.
    echo "status_returned: $status_returned"

    return $status_returned  # Return the status of the command execution.
}


# Log the initial script parameters for reference.
mkdir -p "${OUTPUT_DIR}/run" 
echo "Running script | OUTPUT_DIR: '${OUTPUT_DIR}' | INPUT FILE: '${INPUT_FILES[@]}'" > "${OUTPUT_DIR}/run/status.log" 2>&1
echo "Process LOG Files: ${PROCESS_LOG_FILES} | Process TXT Files: ${PROCESS_TXT_FILES} | Debug Mode: ${DEBUG_MODE} | Environment: ${ENVIRONMENT}" >> "${OUTPUT_DIR}/run/status.log" 2>&1

# Create necessary directories for the pipeline.
mkdir -p "${OUTPUT_DIR}/pandarallel_cache" # Need for pandarallel to work properly for very large data frames
mkdir -p "${OUTPUT_DIR}/test_templates" # Drain template model is saved here
mkdir -p "${OUTPUT_DIR}/log_diagnosis"  # stores the two output reports: ({PRODUCT_NAME}_anomaly.html and {PRODUCT_NAME}_summary.html)
mkdir -p "${OUTPUT_DIR}/developer_debug_files" # metadata stored for debugging
mkdir -p "${OUTPUT_DIR}/metrics" # Stores runtime statistics of the tool
cp -r ./log_diagnosis/templates/libs "${OUTPUT_DIR}/log_diagnosis"  # Copy necessary templates.

# Set environment variables for memory file system and cache directories.
export MEMORY_FS_ROOT="${OUTPUT_DIR}/pandarallel_cache"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$OUTPUT_DIR/cache}"
echo "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"

# Run the first pipeline stage: CMD Args Check
PIPELINE_STAGE="CMDLINE_ARGS_CHECK"
run_pipeline_stage python3 args.py --input_files "${INPUT_FILES[@]}" --output_dir "${OUTPUT_DIR}/log_diagnosis" > "${OUTPUT_DIR}/run/args.log" 2>&1
echo "Command line arguments checked!" >> "${OUTPUT_DIR}/run/status.log" 2>&1

# Proceed only if the previous stage was successful: Log Diagnosis
PIPELINE_STAGE="ANOMALY_DETECTION"
run_pipeline_stage python3 run_log_diagnosis.py --input_files "${INPUT_FILES[@]}" --output_dir "$OUTPUT_DIR" --time_range $TIME_RANGE --debug_mode $DEBUG_MODE --process_log_files $PROCESS_LOG_FILES --process_txt_files $PROCESS_TXT_FILES > "${OUTPUT_DIR}/run/log_diagnosis.log" 2>&1
echo "Anomaly Detection Completed!" >> "${OUTPUT_DIR}/run/status.log" 2>&1

# Causality Python script (currently not supported - v0.2.4).
# cd /log_diagnosis
# PIPELINE_STAGE="CAUSALITY"
# run_pipeline_stage python3 causality.py --input_file "${OUTPUT_DIR}/inferencing_file/inference.csv" --signal_map "${OUTPUT_DIR}/developer_debug_files/temp_id_to_signal_map.json" --output_file "${OUTPUT_DIR}/log_diagnosis/${PRODUCT_NAME}_causality.html" --template_map "${OUTPUT_DIR}/developer_debug_files/temp_id_to_rep_log.json" > "${OUTPUT_DIR}/run/causality.log" 2>&1
# echo "Temporal Causality Completed!" >> "${OUTPUT_DIR}/run/status.log" 2>&1

# Send telemetry to Elasticsearch at the end of the pipeline.
send_stats true


rm -rf "${OUTPUT_DIR}/pandarallel_cache"
rm -rf "${OUTPUT_DIR}/cache"

echo "Log Diagnosis Completed!"
echo "Please check "${OUTPUT_DIR}/run" directory for detailed logs"