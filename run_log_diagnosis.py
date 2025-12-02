import sys
import os

from log_diagnosis.models.model_zero_shot_classifer import ZeroShotModels

# Add the necessary directories to the system path to import custom modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'Drain3')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'log_diagnosis')))

# Importing necessary modules for preprocessing, templatizing, and anomaly detection
from log_diagnosis.models import AllModels, ModelType
from preprocessing.preprocessing import Preprocessing
from Drain3.run_drain.run_drain import Templatizer
from log_diagnosis.anomaly import Anomaly
from argparse import ArgumentParser, ArgumentTypeError
from log_diagnosis.utils import prepare_output_dir


def validate_boolean(value):
    value = value.lower()
    if value not in ('true', 'false'):
        raise ArgumentTypeError(f"Invalid value for boolean argument: {value}. Please use 'True' or 'False'.")
    return value == 'true'

def parse_model_name(value):
    """
    Parse model name argument, accepting both enum values and custom model names.
    
    Args:
        value: String representing either a ZeroShotModels enum name or a model name
        
    Returns:
        ZeroShotModels enum member or the string value itself
    """
    # Try to match against ZeroShotModels enum by name
    try:
        return ZeroShotModels[value.upper()]
    except (KeyError, AttributeError):
        pass
    
    # Try to match against ZeroShotModels enum by value
    for model in ZeroShotModels:
        if model.value == value:
            return model
    
    # If no enum match, return the string as-is (allows custom model names)
    return value

# Import pandarallel for parallel processing capabilities
from pandarallel import pandarallel

if __name__ == "__main__":
    """
    Main script to preprocess log files, generate templates using Drain3, and produce an anomaly report.

    This script performs three main tasks:
    1. Preprocess the log files to clean, format, and temporally sort the log data using the `Preprocessing` class.
    2. Use the Drain3 algorithm (via the `Templatizer` class) to generate log templates from the preprocessed data.
    3. Generate an anomaly report based on the processed log templates using the `Anomaly` class.

    The script accepts several command-line arguments for input files, time range, output directories, 
    and an optional XML file for further analysis.

    Steps:
        1. Parse command-line arguments using `argparse`.
        2. Preprocess the input files using the `Preprocessing` class.
        3. Generate log templates with the `Templatizer` class.
        4. Produce an anomaly report with the `Anomaly` class.

    Command-line Arguments:
        --input_files (list of str): Paths to the input log files to be processed.
        --time_range (str): Time range for which analysis is needed (e.g., all-data).
        --output_dir (str): Path to the directory where output files (reports) will be saved.

    Example usage:
        python run_log_diagnosis.py --input_files file1.txt:file2.txt \
                                    --time_range 'all-data' --output_dir '/path/to/output' \
                                    --debug_mode True
    """

    # Create an argument parser to handle command-line input
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--input_files', nargs="+", type=str, help='Input Files for anomaly report generation')
    arg_parser.add_argument('--time_range', type=str, help='Time range for which analysis needs to be done', default='all-data')
    arg_parser.add_argument('--output_dir', type=str, help='Directory where output will be stored')
    arg_parser.add_argument('--debug_mode', type=str, help='Enable debug mode for saving debug files', default=True)
    arg_parser.add_argument('--process_log_files', type=validate_boolean, help='Flag to indicate if logs should be processed', default=True)
    arg_parser.add_argument('--process_txt_files', type=validate_boolean, help='Flag to indicate if text should be processed', default=False)
    arg_parser.add_argument('--model-type', type=ModelType, help='Type of model to use for the anomaly report (Currently Supported Values: zero_shot)', default=ModelType.ZERO_SHOT)
    arg_parser.add_argument('--model', type=parse_model_name, help='Model to use for the anomaly report (Default: "cross-encoder/nli-MiniLM2-L6-H768")', default=ZeroShotModels.CROSSENCODER)
    arg_parser.add_argument('--clean-up', action='store_true', help='Flag to indicate if the output directory should be cleaned up if it exists')

    # Parse the arguments provided by the user
    args = arg_parser.parse_args()
    print(args)

    prepare_output_dir(args.output_dir, args.clean_up)

    # Step 1: Initialize and run the preprocessing on the input files
    preprocessing_obj = Preprocessing(args.debug_mode)
    preprocessing_obj.preprocess(args.input_files, 
                                 args.time_range, 
                                 args.output_dir,
                                 args.process_log_files,
                                 args.process_txt_files)

    # preprocessing_obj.df columns: "text", "preprocessed_text", "truncated_log", "epoch", "timestamps", "file_names"
    # Step 2: Initialize the Templatizer and create log templates
    drain_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Drain3', 'run_drain', 'drain3.ini'))
    templatizer = Templatizer(debug_mode=args.debug_mode, config_path=drain_config_path)
    templatizer.miner(preprocessing_obj.df, 
                    args.output_dir, 
                    args.output_dir + "/test_templates/tm-test.templates.json")


    # templatizer.df columns: "text", "preprocessed_text", "truncated_log", "epoch", "timestamps", "file_names", "test_ids"
    # Step 3: Initialize the Anomaly detector and generate the anomaly report
    anomaly_obj = Anomaly(args.debug_mode, args.model_type, args.model)
    anomaly_obj.get_anomaly_report(templatizer.df, 
                                   args.output_dir + f"/log_diagnosis/", 
                                   args.output_dir + f"/developer_debug_files/", 
                                   )
