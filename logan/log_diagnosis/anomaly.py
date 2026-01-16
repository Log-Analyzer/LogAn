import os
import json
import pandas as pd
import time
import csv
from .core import Core
from datetime import datetime
from logan.log_diagnosis.utils import get_anomaly_html_str, get_summary_html_str
from logan.log_diagnosis.models import ModelManager, AllModels, ModelType

class Anomaly(Core):
    """
    The Anomaly class is responsible for detecting anomalies in log data dump.
    It processes the log data to identify anomalies, merges similar log windows, and generates reports.
    
    The primary methods include:
        - `find_supersets_and_subsets_`: Merges similar TID tuples by identifying supersets and subsets.
        - `merge_sim_windows`: Merges consecutive log windows with similar characteristics.
        - `epoch_to_str`: Converts epoch timestamps to human-readable strings.
        - `get_anomaly_report`: Generates an anomaly report by processing log data and saving the results in HTML format.

    Attributes:
        None (inherits from Core class).
    
    Usage:
        - Instantiate the class and call its methods to process log data and generate anomaly reports.
        - The class leverages machine learning models for GS and FC to perform anomaly detection.
    """
    def __init__(self, debug_mode, type: ModelType, model: AllModels):
        """
        Initializes the Anomaly class and core class.

        Args:
            debug_mode (str): The debug mode to use for the anomaly report.
            type (ModelType): The type of model to use for the anomaly report.
            model (AllModels): The model to use for the anomaly report.
        """
        super().__init__(type, model)
        self.debug_mode = debug_mode
    
    def find_supersets_and_subsets_(self, input_superset_dict):
        """
        Identifies and merges supersets and subsets within a dictionary of error TID tuples and their corresponding elements.
        
        Args:
            input_superset_dict (dict): A dictionary where keys are tuples of error TIDs, and values are lists of related elements.
        
        Returns:
            dict: A dictionary where subsets are merged into supersets, ensuring minimal redundant entries.
        """
        superset_dict = {}

        for error_tid_tuple, elements in input_superset_dict.items():

            # If superset_dict is empty, add the first error TID tuple and elements
            if len(superset_dict) == 0:
                superset_dict[error_tid_tuple] = elements
                continue

            merged_flag = False
            # Iterate through existing supersets to find any that can be merged
            for key_tup in list(superset_dict.keys()):
                superset = set(key_tup)

                # Check if the current error_tid_tuple is a subset or superset of an existing key
                if superset.issuperset(set(error_tid_tuple)):
                    superset_dict[key_tup].extend(elements)
                    merged_flag = True
                    break
                elif set(error_tid_tuple).issuperset(superset):
                    superset_dict[error_tid_tuple] = superset_dict.pop(key_tup)
                    superset_dict[error_tid_tuple].extend(elements)
                    merged_flag = True
                    break

            # If no merging occurred, add the new entry to the dictionary
            if not merged_flag:
                try:
                    if elements not in superset_dict[error_tid_tuple]:
                        superset_dict[error_tid_tuple] = elements
                except KeyError:
                    superset_dict[error_tid_tuple] = elements

        # If input_superset_dict has different size than superset_dict, repeat the merging process
        if len(input_superset_dict) != len(superset_dict):
            superset_dict = self.find_supersets_and_subsets_(superset_dict)
        
        return superset_dict

    def merge_sim_windows(self, df):
        """
        Merges similar windows (log segments) from a DataFrame by finding supersets and combining logs, file names, and TIDs.

        Args:
            df (pd.DataFrame): Input DataFrame with columns 'epoch', 'text_output', 'file_names', 'error_test_ids', and 'test_ids'.
        
        Returns:
            pd.DataFrame: A DataFrame where similar windows have been merged, containing columns for start and end epochs, logs, file names, and TIDs.
        """
        # Extract necessary data from DataFrame columns into lists
        epochs = df["epoch"].to_list()
        logs_lists = [logs_string.split("\n") for logs_string in df["text_output"].to_list()]
        file_names_lists = [file_names_string.split("\n") for file_names_string in df["file_names"].to_list()]
        error_tids_lists = [[tid for tid in test_ids.split(" ") if tid != "info"] for test_ids in df["error_test_ids"].to_list()]
        tids_lists = [test_ids.split(" ") for test_ids in df["test_ids"].to_list()]

        # Initialize superset dictionary with tuples of error TIDs
        superset_dict = {tuple(error_tids_list): [(tids_list, logs_list, file_names_list, epoch, error_tids_list)] 
                         for tids_list, logs_list, file_names_list, epoch, error_tids_list 
                         in zip(tids_lists, logs_lists, file_names_lists, epochs, error_tids_lists)}

        # Find and merge supersets and subsets within the data
        superset_dict = self.find_supersets_and_subsets_(superset_dict)

        # Prepare lists to store merged results
        tid_str_list = []
        error_tid_list = []
        log_str_list = []
        file_str_list = []
        start_epoch_list = []
        end_epoch_list = []

        for _, elements in superset_dict.items():
            # Sort elements by epoch to ensure chronological order
            elements = sorted(elements, key=lambda x: x[3])

            tids, logs, file_names, error_tids = [], [], [], []

            # Merge TIDs, logs, and file names from all elements in the superset
            [tids.extend(sublist[0]) for sublist in elements]
            [logs.extend(sublist[1]) for sublist in elements]
            [file_names.extend(sublist[2]) for sublist in elements]
            [error_tids.extend(sublist[4]) for sublist in elements]

            # Build concatenated strings for TIDs, logs, and file names
            first_indices = [(item, index) for index, item in enumerate(tids)]

            tid_str, log_str, file_str = "", "", ""
            for tid, index in first_indices:
                tid_str += " " + tid
                log_str += "\n" + logs[index]
                file_str += "\n" + file_names[index]

            # Remove duplicates and format the strings
            error_tids = set(error_tids)
            error_tid_str = " ".join(error_tids)

            tid_str = tid_str.strip()
            error_tid_str = error_tid_str.strip()
            log_str = log_str.strip()
            file_str = file_str.strip()

            # Define start and end epochs for the merged window
            start_epoch = elements[0][3]
            end_epoch = elements[-1][3] + 30  # Add buffer time to end epoch
            
            # Append results to corresponding lists
            tid_str_list.append(tid_str)
            error_tid_list.append(error_tid_str)
            log_str_list.append(log_str)
            file_str_list.append(file_str)
            start_epoch_list.append(start_epoch)
            end_epoch_list.append(end_epoch)

        # Create output DataFrame with merged windows
        df_output = pd.DataFrame.from_dict({
            "start_epoch": start_epoch_list,
            "end_epoch": end_epoch_list,
            "text_output": log_str_list,
            "file_names": file_str_list,
            "test_ids": tid_str_list,
            "error_test_ids": error_tid_list
        })

        return df_output

    def epoch_to_str(self, epoch):
        """
        Converts an epoch timestamp to a human-readable GMT time string.
        
        Args:
            epoch (int): Epoch time in seconds.
        
        Returns:
            str: A string representing the GMT time in 'YYYY-MM-DD HH:MM:SS' format.
        """
        # Convert epoch to UTC datetime object
        gmt_time = datetime.utcfromtimestamp(epoch)

        # Format datetime object as a string in GMT timezone
        gmt_time_string = gmt_time.strftime('%Y-%m-%d %H:%M:%S')

        return gmt_time_string

    def compute_anomaly_statistics(self, output_dir, time):
        with open(os.path.join(output_dir, "metrics", "anomaly.json"), 'w') as writer:
            data = {
                'anomaly_detection_time_ms': time,
            }
            writer.write(json.dumps(data, indent=4))

    def get_anomaly_report(self, df_inference_csv, output_dir):
        """
        Generates an anomaly report from the input DataFrame, processes it for anomalies, merges similar windows,
        and saves the results in HTML format for both anomalies and summary reports.
        
        Args:
            df_inference_csv (pd.DataFrame): The input data containing log information to process.
            output_file (str): Output file path for saving the anomaly HTML report.
            debug_file (str): Debug file path for storing intermediate files.
        
        Returns:
            None
        """
        print("Output directory:", output_dir)

        log_diagnosis_dir = os.path.join(output_dir, "log_diagnosis")
        developer_debug_dir = os.path.join(output_dir, "developer_debug_files")

        start = time.time()

        # Disable token parallelism to prevent concurrency issues
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Model paths and batch size settings
        MODEL_GS = "../llm/model_gs"
        MODEL_FAULT_CATEGORY = "../llm/model_fault"
        BATCH_SIZE = 64

        start_time_post = time.time()

        # Drop any rows with missing data
        df_inference_csv = df_inference_csv.dropna()
        print(f"len of df_inference_csv after 1st read: {len(df_inference_csv)}")
        print("starting the preprocessing of input data")

        # Process data to detect anomalies and generate HTML dataframes
        df_for_anomaly_html, df_for_summary_html, temp_id_to_signal_map, temp_id_to_rep_log, df_for_anomaly_html_non_info = self.process_data(
            df_inference_csv, MODEL_GS, MODEL_FAULT_CATEGORY, BATCH_SIZE
        )
        print("ended the preprocessing of input data")
        
        # Log template and signal map debug files
        debug_file_path = os.path.join(developer_debug_dir, "temp_id_to_rep_log.json")  
        if (self.debug_mode == "true"):
            with open(debug_file_path, 'w') as writer:
                json.dump(temp_id_to_rep_log, writer, indent=4) 
        json_serializable_map = {str(k): v for k, v in temp_id_to_signal_map.items()}
        debug_file_path = os.path.join(developer_debug_dir, "temp_id_to_signal_map.json") 
        if (self.debug_mode == "true"):
            with open(debug_file_path, 'w') as writer:
                json.dump(json_serializable_map, writer, indent=4)
        
        # Process logs with only 'information' golden signals
        all_info_df = df_for_anomaly_html[df_for_anomaly_html.all_info == True]
        start_ts_all_info = all_info_df["epoch"].tolist()
        template_ids2_all_info = all_info_df["test_ids"].tolist()
        log_seq_all_info = all_info_df["text_output"].tolist()

        # Convert start and end epochs to readable strings
        end_ts_all_info = [self.epoch_to_str(ts + 30) for ts in start_ts_all_info]
        start_ts_all_info = [self.epoch_to_str(ts) for ts in start_ts_all_info]

        # Create a DataFrame for the 'all information' windows
        df_all_info = pd.DataFrame.from_dict({
            'start_ts': start_ts_all_info,
            'end_ts': end_ts_all_info,
            'list_logs': log_seq_all_info,
            'list_templates': template_ids2_all_info
        })

        normal_win_count = len(log_seq_all_info)

        # Process logs for anomalies (non-information golden signals)
        df_final_anomalies = df_for_anomaly_html[df_for_anomaly_html.all_info != True]
        df_final_anomalies = self.merge_sim_windows(df_final_anomalies)

        # Extract and sort start/end times, logs, file names, and templates for the anomalies
        start_ts = df_final_anomalies["start_epoch"].tolist()
        end_ts = df_final_anomalies["end_epoch"].tolist()
        template_ids2 = df_final_anomalies["test_ids"].tolist()
        log_seq = df_final_anomalies["text_output"].tolist()
        file_seq = df_final_anomalies["file_names"].tolist()
        if len(log_seq) != 0:
            start_ts, end_ts, template_ids2, log_seq, file_seq = zip(*sorted(zip(start_ts, end_ts, template_ids2, log_seq, file_seq), reverse=True))
            end_ts = [self.epoch_to_str(ts) for ts in end_ts]
            start_ts = [self.epoch_to_str(ts) for ts in start_ts]

        print(f"total normal windows: {normal_win_count}")
        print(f"total anomalous windows: {len(log_seq)}")  
        print(f"time in seconds for complete post request: {time.time() - start_time_post}")
        
        # Create a DataFrame for the final anomaly results
        df_final_anomalies = pd.DataFrame.from_dict({
            'start_ts': start_ts,
            'end_ts': end_ts,
            'list_logs': log_seq,
            'list_files': file_seq,
            'list_result': ["Anomaly"]*len(log_seq),
            'list_templates': template_ids2
        })

        # Read debug information (ignored and processed files)
        with open(os.path.join(developer_debug_dir, "ignored_files.log"), 'r') as reader:
            ignored_files = reader.read().splitlines()

        with open(os.path.join(developer_debug_dir, "processed_files.log"), 'r') as reader:
            processsed_files = reader.read().splitlines()
        
        # Generate the HTML table for the anomaly report
        html_table = get_anomaly_html_str(df_final_anomalies, output_dir)
        with open(os.path.join(log_diagnosis_dir, "anomalies.html"), "w") as f:
            f.write(html_table)

        # Generate the HTML table for the summary report
        html_table = get_summary_html_str(df_for_summary_html, include_golden_signal_dropdown=True, ignored_file_list=ignored_files, processed_file_list=processsed_files)
        with open(os.path.join(log_diagnosis_dir, "summary.html"), "w") as f:
            f.write(html_table)

        self.compute_anomaly_statistics(output_dir, (time.time() - start) * 1000)
