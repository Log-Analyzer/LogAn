set -euo pipefail
export IMG=aiops-log-diagnosis:latest

ENV=podman
# ENV=docker

build() {
    chmod +x run.sh
    # DOCKER_BUILDKIT=0 $ENV build --build-arg ARTIFACTORY_USERNAME=${ARTIFACTORY_USERNAME} --build-arg ARTIFACTORY_API_KEY=${ARTIFACTORY_API_KEY} . -t ${IMG}
    DOCKER_BUILDKIT=0 $ENV build . -t ${IMG}
}

handle_path_space_and_escape() {
    path="$1"
    realpath_quoted=$(realpath "$path")
    # escaped_path=$(printf '%q' "$realpath_quoted")
    
    echo "\"$realpath_quoted\""
}



run(){
    ALL_ARGS="[ OUTPUT_DIR LOG_FILE_PATH TIME_RANGE(OPTIONAL) -ProcessLogFiles(OPTIONAL) -ProcessTxtFiles(OPTIONAL) -DebugMode(OPTIONAL) ]"

    if [ "$#" -lt 3 ] ||  [ "$#" -gt 8 ]; then
        echo "Please provide all the arguments in this order: $ALL_ARGS"
        exit 1
    fi
    
    mkdir -p "$1"
    OUTPUT_DIR=$(handle_path_space_and_escape "$1")

    # PRODUCT_NAME=$2

    delimiter=':'
    IFS=$delimiter read -r -a FILENAMES <<< "$2"

    # Array of optional flags (skip first three required arguments)
    shift 2
    OPTIONS=("$@")

    # Initialize optional variables
    TIME_RANGE="all-data"
    PROCESS_TXT_FILES=false
    DEBUG_MODE=true
    ## If you don't want to process .log files, then set the below variable to `false`.
    ## `docker.sh` doesn't provide options which will set the value `PROCESS_LOG_FILES` as `false`.
    ## Specifically, the following flags are NOT SUPPORTED:  `-NoProcessLogFiles` or `-ProcessLogFiles=false`
    PROCESS_LOG_FILES=true


    for variable in "${OPTIONS[@]}"; do
        case $variable in
            1-day | 2-day | 3-day | 4-day | 5-day | 6-day | 1-week | 2-week | 3-week | 1-month | all-data )
                TIME_RANGE=$variable
                ;;
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
                echo "Error: Unknown Flag - $variable"
                echo "Allowed Values: $ALL_ARGS"
                exit 1
                ;;
        esac
    done

    echo "Initializing AnalyzeSWSupport with the following parameters:"
    echo
    echo "OUTPUT_DIR: $OUTPUT_DIR"
    echo "LOG_FILE_PATH: ${FILENAMES[@]}"
    echo "TIME_RANGE: $TIME_RANGE"
    echo "-ProcessLogFiles: $PROCESS_LOG_FILES"
    echo "-ProcessTxtFiles: $PROCESS_TXT_FILES"
    echo "-DebugMode: $DEBUG_MODE"
    echo

    docker_command="$ENV run --userns=keep-id --user $(id -u):$(id -g) --shm-size=150g"
    file_environment_variable=""
    mount_index=0

    for filename in "${FILENAMES[@]}"; do
        # Perform tilde expansion for each element
        # eval "filename=${filenames[$i]}"

        filename_with_space_handled=$(handle_path_space_and_escape "$filename")
        
        # Create target path inside container under /mnt/logs/
        target_path="/mnt/logs/$(basename "$filename")"
        
        # If multiple files have the same basename, append index
        while [[ "$file_environment_variable" == *"$target_path"* ]]; do
            mount_index=$((mount_index + 1))
            target_path="/mnt/logs/$(basename "$filename")_$mount_index"
        done

        file_environment_variable+="$target_path$delimiter"
        docker_command+=" --mount src=$filename_with_space_handled,target=$target_path,type=bind,Z"
    done
    # file_environment_variable="${file_environment_variable%$delimiter}"
    docker_command+=" --mount src="$OUTPUT_DIR",target=/mnt/output,type=bind,Z"
    docker_command+=" -e OUT_DIRECTORY=/mnt/output"
    docker_command+=" -e FILE=/mnt/logs/"

    EXTRA_ARGS="TimeRange ${TIME_RANGE}"
    if [ "$PROCESS_LOG_FILES" = "true" ]; then
        EXTRA_ARGS+=" -ProcessLogFiles"
    fi
    if [ "$PROCESS_TXT_FILES" = "true" ]; then
        EXTRA_ARGS+=" -ProcessTxtFiles"
    fi
    if [ "$DEBUG_MODE" = "true" ]; then 
        EXTRA_ARGS+=" -DebugMode"
    fi

    EXTRA_ARGS+=" Enviroment LOCAL"

    docker_command+=" -e EXTRA_ARGS=\"$EXTRA_ARGS\""
    
    # Mount the transformers cache - ensure the directory exists and has proper permissions
    export HF_HOME="~/.cache/huggingface"
    export MOUNT_HF_HOME="/opt/.cache/huggingface"
    docker_command+=" -e HF_HOME=$MOUNT_HF_HOME"
    docker_command+=" --mount src=$HF_HOME,target=$MOUNT_HF_HOME,type=bind,Z"

    docker_command+=" --rm -it ${IMG}"
    
    echo 
    echo $docker_command
    echo
    eval "$docker_command"
}

scan() {
    mkdir tmp2
    $ENV run --privileged -v /var/run/docker.sock:/var/run/docker.sock -v ${PWD}/tmp2:/tmp aquasec/trivy image --format template --template "@contrib/html.tpl" -o /tmp/report.html --timeout 10m0s ${IMG}
}

test() {
    echo "Running tests directly via bash..."
    if ! command -v pytest &> /dev/null; then
        echo "pytest not found. Please install pytest first."
        exit 1
    fi
    cd preprocessing
    pytest -v test_preprocessing.py
    cd ..
}

"$@"