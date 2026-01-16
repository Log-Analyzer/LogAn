#!/bin/bash
#
# Logan Container Entrypoint Script
# 
# This script wraps the Logan CLI tool and allows configuration via environment variables.
# It supports two modes: 'analyze' for log analysis and 'view' for serving reports.
#

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

#######################################
# Environment Variable Defaults
#######################################

# Mode: 'analyze' to run log analysis, 'view' to serve report via HTTP
LOGAN_MODE="${LOGAN_MODE:-analyze}"

# Input files/directories (comma-separated for multiple)
LOGAN_INPUT_FILES="${LOGAN_INPUT_FILES:-/data/input}"

# Output directory for analysis results
LOGAN_OUTPUT_DIR="${LOGAN_OUTPUT_DIR:-/data/output}"

# Time range for analysis (all-data, 1-day, 2-day, ..., 1-week, 2-week, 1-month)
LOGAN_TIME_RANGE="${LOGAN_TIME_RANGE:-all-data}"

# Model type for anomaly detection (zero_shot, similarity, custom)
LOGAN_MODEL_TYPE="${LOGAN_MODEL_TYPE:-zero_shot}"

# Model to use for classification (bart, crossencoder, or custom HuggingFace model)
LOGAN_MODEL="${LOGAN_MODEL:-crossencoder}"

# Enable debug mode (saves debug files)
LOGAN_DEBUG_MODE="${LOGAN_DEBUG_MODE:-true}"

# Process .log files from directories
LOGAN_PROCESS_LOG_FILES="${LOGAN_PROCESS_LOG_FILES:-true}"

# Process .txt files from directories
LOGAN_PROCESS_TXT_FILES="${LOGAN_PROCESS_TXT_FILES:-false}"

# Clean up output directory before running
LOGAN_CLEAN_UP="${LOGAN_CLEAN_UP:-false}"

# Port for view mode HTTP server
LOGAN_VIEW_PORT="${LOGAN_VIEW_PORT:-8000}"

# Directory to serve in view mode (defaults to LOGAN_OUTPUT_DIR if not set)
LOGAN_VIEW_DIR="${LOGAN_VIEW_DIR:-}"

#######################################

# Determine the command prefix
# If running in a uv-managed venv (has .venv/bin/logan), use that directly
# Otherwise fall back to system logan command
if [ -x ".venv/bin/logan" ]; then
    LOGAN_CMD=".venv/bin/logan"
elif [ -x "/opt/app-root/src/.venv/bin/logan" ]; then
    # Container path
    LOGAN_CMD="/opt/app-root/src/.venv/bin/logan"
elif command -v logan &> /dev/null; then
    LOGAN_CMD="logan"
else
    echo -e "${RED}Error: logan command not found${NC}"
    echo "Make sure logan is installed: pip install -e . or uv pip install -e ."
    exit 1
fi

echo -e "${CYAN}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║           Logan - Log Analysis Tool                       ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Print current configuration
print_config() {
    echo -e "${YELLOW}Current Configuration:${NC}"
    echo "  LOGAN_CMD:               ${LOGAN_CMD}"
    echo "  LOGAN_MODE:              ${LOGAN_MODE}"
    echo "  LOGAN_INPUT_FILES:       ${LOGAN_INPUT_FILES}"
    echo "  LOGAN_OUTPUT_DIR:        ${LOGAN_OUTPUT_DIR}"
    echo "  LOGAN_TIME_RANGE:        ${LOGAN_TIME_RANGE}"
    echo "  LOGAN_MODEL_TYPE:        ${LOGAN_MODEL_TYPE}"
    echo "  LOGAN_MODEL:             ${LOGAN_MODEL}"
    echo "  LOGAN_DEBUG_MODE:        ${LOGAN_DEBUG_MODE}"
    echo "  LOGAN_PROCESS_LOG_FILES: ${LOGAN_PROCESS_LOG_FILES}"
    echo "  LOGAN_PROCESS_TXT_FILES: ${LOGAN_PROCESS_TXT_FILES}"
    echo "  LOGAN_CLEAN_UP:          ${LOGAN_CLEAN_UP}"
    echo "  LOGAN_VIEW_PORT:         ${LOGAN_VIEW_PORT}"
    echo ""
}

# Run analyze mode
run_analyze() {
    echo -e "${GREEN}Running Logan in ANALYZE mode...${NC}"
    echo ""

    # Validate required environment variables
    if [ -z "${LOGAN_INPUT_FILES}" ]; then
        echo -e "${RED}Error: LOGAN_INPUT_FILES is required for analyze mode${NC}"
        echo "Please set LOGAN_INPUT_FILES to a comma-separated list of files or directories"
        exit 1
    fi

    if [ -z "${LOGAN_OUTPUT_DIR}" ]; then
        echo -e "${RED}Error: LOGAN_OUTPUT_DIR is required for analyze mode${NC}"
        exit 1
    fi

    # Build the command
    CMD="${LOGAN_CMD} analyze"

    # Parse comma-separated input files and add each as a -f flag
    IFS=',' read -ra FILES <<< "${LOGAN_INPUT_FILES}"
    for file in "${FILES[@]}"; do
        # Trim whitespace
        file=$(echo "$file" | xargs)
        if [ -n "$file" ]; then
            CMD="$CMD -f \"$file\""
        fi
    done

    # Add output directory
    CMD="$CMD -o \"${LOGAN_OUTPUT_DIR}\""

    # Add time range
    CMD="$CMD -t ${LOGAN_TIME_RANGE}"

    # Add model type
    CMD="$CMD --model-type ${LOGAN_MODEL_TYPE}"

    # Add model
    CMD="$CMD -m ${LOGAN_MODEL}"

    # Add debug mode flag
    if [ "${LOGAN_DEBUG_MODE,,}" = "true" ]; then
        CMD="$CMD --debug-mode"
    else
        CMD="$CMD --no-debug-mode"
    fi

    # Add process log files flag
    if [ "${LOGAN_PROCESS_LOG_FILES,,}" = "true" ]; then
        CMD="$CMD --process-log-files"
    else
        CMD="$CMD --no-process-log-files"
    fi

    # Add process txt files flag
    if [ "${LOGAN_PROCESS_TXT_FILES,,}" = "true" ]; then
        CMD="$CMD --process-txt-files"
    else
        CMD="$CMD --no-process-txt-files"
    fi

    # Add clean up flag
    if [ "${LOGAN_CLEAN_UP,,}" = "true" ]; then
        CMD="$CMD --clean-up"
    fi

    echo -e "${CYAN}Executing: ${CMD}${NC}"
    echo ""

    # Execute the command
    eval $CMD

    # Make output directory contents accessible to all users (for mounted volumes)
    echo -e "${YELLOW}Setting permissions on output directory...${NC}"
    chmod -R 777 "${LOGAN_OUTPUT_DIR}" 2>/dev/null || true
}

# Run view mode
run_view() {
    echo -e "${GREEN}Running Logan in VIEW mode...${NC}"
    echo ""

    # Use output dir as view dir if LOGAN_VIEW_DIR is not set
    VIEW_DIR="${LOGAN_VIEW_DIR:-${LOGAN_OUTPUT_DIR}}"

    if [ -z "${VIEW_DIR}" ]; then
        echo -e "${RED}Error: LOGAN_VIEW_DIR or LOGAN_OUTPUT_DIR is required for view mode${NC}"
        exit 1
    fi

    CMD="${LOGAN_CMD} view -d \"${VIEW_DIR}\" -p ${LOGAN_VIEW_PORT}"

    echo -e "${CYAN}Executing: ${CMD}${NC}"
    echo ""
    echo -e "${YELLOW}Starting web server on port ${LOGAN_VIEW_PORT}...${NC}"
    echo -e "${YELLOW}Access the report at: http://localhost:${LOGAN_VIEW_PORT}/${VIEW_DIR}/log_diagnosis/${NC}"
    echo ""

    # Execute the command
    eval $CMD
}

# Main execution
print_config

case "${LOGAN_MODE}" in
    analyze)
        run_analyze
        ;;
    view)
        run_view
        ;;
    *)
        echo -e "${RED}Error: Invalid LOGAN_MODE '${LOGAN_MODE}'${NC}"
        echo ""
        echo "Supported modes:"
        echo "  analyze - Analyze log files for anomalies"
        echo "  view    - Start web server to view analysis reports"
        echo ""
        echo "Usage:"
        echo "  Set LOGAN_MODE=analyze or LOGAN_MODE=view"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Done!${NC}"

