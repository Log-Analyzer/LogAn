import os
from datetime import datetime, timezone, timedelta
import time
import numpy as np
from dateutil import parser as god_parse
from dateutil import tz
import pandas as pd
import gc
import statistics
from tqdm import tqdm
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset

from logan.log_diagnosis.models import ModelManager, AllModels, ModelType

tqdm.pandas()
np.random.seed(42)

class Core:
    """
    The Core class provides methods for processing log data, detecting golden signals, fault categories, 
    and generate dataframes for generation of anomaly and summary report based on the processed logs.
    
    Uses zero-shot classification to categorize logs into: Info, Error, Network, Availability.

    Methods:
        backprop_gs_fault_with_temp_ids(row, mapping):
            Determines the golden signal and fault categories for each log entry based on template IDs and mapping.

        convert_to_epoch(timestamp_str, format):
            Converts a timestamp string to an epoch timestamp.

        epoch_to_str(epoch, format):
            Converts an epoch timestamp to a formatted string.

        get_fault(list_logs, model_fault, batch_size):
            Detects fault categories in the provided log data using zero-shot classification.

        multi_rep_weighted_output(final_outputs_per_rep, final_scores_per_rep):
            predictions of GS signal for multiple reps of a template ID are used to determine the final signal. Predictions are weighed using the predicted probability.

        get_gs(list_logs, model_gs, batch_size):
            Detects golden signals in the provided log data using zero-shot classification.

        select_first_item(lst):
            Selects first item from a list of logs and returns it in a list

        process_data(df_inference_csv, model_gs, model_fault, batch_size):
            Processes the input DataFrame to detect golden signals and fault categories using zero-shot classification, and generates dataframe for summary and anomaly reports.
    """
    
    def __init__(self, type: ModelType, model: AllModels):
        """
        Initializes the Core class and model manager.
        """
        self.model_manager = ModelManager(type, model)

    def backprop_gs_fault_with_temp_ids(self, row, mapping):
        """
        Determines the golden signal and fault categories for a log entry based on its template ID and file name.
        
        Args:
            row (pd.Series): A row from the DataFrame containing the log details.
            mapping (dict): A mapping of (template_id, file_name) to (golden_signal, fault_categories).

        Returns:
            pd.Series: A series containing the golden signal and the formatted log string with fault categories.
        """
        templateId_new = row["test_ids"]
        file_name = row["file_names"]
        log = row["text"]
        
        if templateId_new == -1:
            gs, fault = "Info", [["other"]]
        else:
            gs, fault = mapping[(templateId_new, file_name)]

        log = log.replace("\n", "&#13;&#10;")
        out_string = f"{log} => Fault-Categories: {fault[0]} => Golden-Signal: {gs}"
        return pd.Series([gs, out_string])
    
    def convert_to_epoch(self, timestamp_str, format):
        """
        Converts a timestamp string to an epoch timestamp.
        
        Args:
            timestamp_str (str): The timestamp string to convert.
            format (str): The format of the timestamp string (not used in this method).

        Returns:
            int: The epoch timestamp.
        """
        try:
            tzinfos = {"CDT": tz.gettz("America/Chicago")}  # Map "CDT" to Central Daylight Time
            parsed_date = god_parse.parse(timestamp_str, tzinfos=tzinfos)
            epoch_time = int(parsed_date.timestamp())
            return epoch_time
        except ValueError as e:
            print("Error: ", e)
            return None
        
    def epoch_to_str(self, epoch, format):
        """
        Converts an epoch timestamp to a formatted string.
        
        Args:
            epoch (int): The epoch timestamp to convert.
            format (str): The format for the resulting string.

        Returns:
            str: The formatted date-time string.
        """
        dt_object = datetime.fromtimestamp(epoch, tz=timezone.utc)
        gmt_offset = timedelta(hours=5, minutes=30)  # Offset for IST (Indian Standard Time)
        dt_object = dt_object + gmt_offset
        time_string = dt_object.strftime(format)
        return time_string

    def get_fault(self, list_logs, model_fault=None, batch_size=32):
        """
        Detects fault categories in the provided logs using zero-shot classification.
        
        Args:
            list_logs (list of lists): A list of lists, where each sublist contains log entries.
            model_fault (str): Unused parameter (kept for compatibility).
            batch_size (int): The batch size for processing logs.

        Returns:
            list of lists: A list of fault categories corresponding to the input logs.
        """
        
        index = []
        start_idx = 0
        end_ix = 0
        list_of_logs = []
        for i, logs in enumerate(list_logs):
            end_ix += len(logs)
            index.append((start_idx, end_ix))
            start_idx = end_ix
            list_of_logs.extend(logs)

        # Initialize zero-shot classification pipeline
        predictions = self.model_manager.classify_fault_category(list_of_logs, batch_size)

        # Extract predictions with confidence threshold
        fault_predictions = []
        for pred in predictions:
            # Get all labels with score > 0.3 (adjustable threshold)
            high_conf_labels = [label for label, score in zip(pred['labels'], pred['scores']) if score > 0.2]
            # If no high confidence labels, take the top prediction
            if not high_conf_labels:
                high_conf_labels = [pred['labels'][0]]
            fault_predictions.append(high_conf_labels)

        final_output = [fault_predictions[start:end] for start, end in index]
        del predictions, fault_predictions, list_of_logs

        return final_output

    def multi_rep_weighted_output(self, final_outputs_per_rep, final_scores_per_rep):
        """
        predictions of GS signal for multiple reps of a template ID are used to determine the final signal. Predictions are weighed using the predicted probability.
        
        Args:
            final_outputs_per_rep (list of lists): List of predicted classes from different reps of a template ID if representations.
            final_scores_per_rep (list of lists): List of prediction scores corresponding to the predictions from different reps of a template ID if representations.

        Returns:
            list: The final class for each log entry, determined by weighted average and tie-breaking rules.
        """
        final_output = []
        
        for fopr, fcpr in zip(final_outputs_per_rep, final_scores_per_rep):
            # Combine the predicted classes and scores into a dictionary
            class_scores = {}
            for cls, score in zip(fopr, fcpr):
                class_scores.setdefault(cls, {'sum': 0, 'count': 0, 'score': []})
                class_scores[cls]['sum'] += score
                class_scores[cls]['count'] += 1
                class_scores[cls]['score'].append(score)
            
            # Calculate the average score for each class
            class_avg_scores = {cls: statistics.mean(values['score']) for cls, values in class_scores.items()}
            
            # Find the class with the highest average score
            max_avg_score = max(class_avg_scores.values())
            best_classes = [cls for cls, avg_score in class_avg_scores.items() if avg_score == max_avg_score]
            
            # If there are ties in average score, find the class with the highest count
            if len(best_classes) > 1:
                max_count = max(class_scores[cls]['count'] for cls in best_classes)
                best_classes = [cls for cls in best_classes if class_scores[cls]['count'] == max_count]
            
            # If there are still ties, randomly select a class
            final_class = np.random.choice(best_classes)

            final_output.append(final_class)
        return final_output

    def get_gs(self, list_logs, model_gs=None, batch_size=32):
        """
        Detects golden signals in the provided logs using zero-shot classification.
        
        Args:
            list_logs (list of lists): A list of lists, where each sublist contains log entries.
            model_gs (str): Unused parameter (kept for compatibility).
            batch_size (int): The batch size for processing logs.

        Returns:
            list: The detected golden signals for each list of logs.
        """
        # Define candidate labels for zero-shot classification
        candidate_labels = ["information", "error", "availability", "latency", "saturation", "traffic"]
        
        # This block flattens the nested list of logs (list_logs) into a single list (list_of_logs),
        # while also keeping track of the start and end indices for each original sublist.
        # The 'index' list stores tuples (start_idx, end_ix) for each sublist, so that after
        # processing, results can be mapped back to the original structure.
        index = []
        start_idx = 0
        end_ix = 0
        list_of_logs = []
        for logs in list_logs:
            end_ix += len(logs)
            index.append((start_idx, end_ix))
            start_idx = end_ix
            list_of_logs.extend(logs)

        # Initialize zero-shot classification pipeline
        predictions = self.model_manager.classify_golden_signal(list_of_logs, batch_size)
        print(f"Predictions: {predictions[0]}")

        output = [pred['labels'][0] for pred in predictions]
        y_scores = [pred['scores'][0] for pred in predictions]
        del predictions, list_of_logs

        final_outputs_per_rep = [output[start:end] for start, end in index]
        final_scores_per_rep = [y_scores[start:end] for start, end in index]
        
        final_output = self.multi_rep_weighted_output(final_outputs_per_rep, final_scores_per_rep)
        
        return final_output

    def select_first_item(self, lst):
        return [lst.iloc[0]]

    def process_data(self, df_inference_csv, model_gs, model_fault, batch_size):
        """
        Processes the input DataFrame to detect golden signals and fault categories, and generates summary and anomaly reports.
        
        Args:
            df_inference_csv (pd.DataFrame): The input DataFrame containing log data.
            model_gs (str): The path to the model used for golden signal detection.
            model_fault (str): The path to the model used for fault detection.
            batch_size (int): The batch size for processing logs.

        Returns:
            tuple: A tuple containing:
                - pd.DataFrame: DataFrame for anomaly reports.
                - pd.DataFrame: DataFrame for summary reports.
                - dict: A mapping of (test_id, file_name) to (golden_signal, fault_categories).
                - dict: A mapping of (test_id, file_name) to representative logs.
                - pd.DataFrame: DataFrame for anomaly reports excluding 'information' golden signals.
        """
        print(f"MEMORY_FS_ROOT: {os.environ['MEMORY_FS_ROOT']}")
        
        if df_inference_csv.empty:
            # Create empty DataFrames with the necessary columns for early exit
            columns_anomaly = ['group', 'test_ids', 'error_test_ids', 'file_names', 'text_output', 'epoch', 'golden_signal', 'all_info', 'len']
            columns_summary = ['d_tid', 'gs', 'd_tid_count', 'text', 'file_names', 'coverage']
            
            df_for_anomaly_html = pd.DataFrame(columns=columns_anomaly)
            df_for_summary_html = pd.DataFrame(columns=columns_summary)
            df_for_anomaly_html_non_info = pd.DataFrame(columns=columns_anomaly[:-2] + ['len'])
            
            return df_for_anomaly_html, df_for_summary_html, {}, {}, df_for_anomaly_html_non_info

        # Grouping and preprocessing for representative logs
        representative_df = df_inference_csv.groupby(['test_ids', 'file_names']).agg({
            'preprocessed_text': self.select_first_item,
            'text': self.select_first_item,
            'timestamps': self.select_first_item
        }).reset_index()

        rep_lol = representative_df["preprocessed_text"].tolist()
        temp_ids = representative_df["test_ids"].tolist()
        file_names = representative_df["file_names"].tolist()
        del representative_df

        # Detecting Golden Signal
        print("Detecting Golden Signal")
        start_time = time.time()
        gs_list = self.get_gs(rep_lol, model_gs, batch_size)
        print(f"Golden signal detection completed in: {time.time() - start_time} seconds")

        # Detecting Fault Categories
        print("Detecting Fault Categories")
        start_time = time.time()
        logs_for_fcp = [lol for gs, lol in zip(gs_list, rep_lol) if gs != 'Info']
        fault_list = self.get_fault(logs_for_fcp, model_fault, batch_size)
        print(f"Fault category detection completed in: {time.time() - start_time} seconds")

        # Free model memory — models are no longer needed after inference
        del self.model_manager
        gc.collect()

        # Mapping test IDs to their corresponding golden signal and fault category
        temp_id_to_signal_map = {}
        fcp_idx = 0
        for idx, (tid, file_name) in enumerate(zip(temp_ids, file_names)):
            gs = gs_list[idx]
            if gs == 'Info':
                temp_id_to_signal_map[(tid, file_name)] = ('Info', ['other'])
            else:
                temp_id_to_signal_map[(tid, file_name)] = (gs, fault_list[fcp_idx])
                fcp_idx += 1

        temp_id_to_rep_log = {tid: {file_name: logs[0]} for tid, file_name, logs in zip(temp_ids, file_names, rep_lol)}
        del rep_lol, temp_ids, file_names, gs_list, fault_list, logs_for_fcp

        # Backtracking GS and Fault Labels
        print("Backtracking GS and Fault Labels")
        start_time = time.time()
        apply_func = self.backprop_gs_fault_with_temp_ids
        df_inference_csv[['golden_signal', 'text_output']] = df_inference_csv.progress_apply(apply_func, axis=1, mapping=temp_id_to_signal_map)
        print(f"Backtracking completed in: {time.time() - start_time} seconds")

        df_inference_csv['test_ids'] = df_inference_csv['test_ids'].astype(str)
        df_inference_csv['error_test_ids'] = df_inference_csv.apply(lambda row: row['test_ids'] if row['golden_signal'] != 'Info' else 'info', axis=1)

        df_inference_csv_only_non_info = df_inference_csv[df_inference_csv['golden_signal'] != "Info"].copy()
        print("Data Frame Only Non-Information is as follows=>")
        print(df_inference_csv_only_non_info.head())

        # Creating Summary DataFrame
        df_for_summary_html = df_inference_csv.groupby(['test_ids', 'file_names']).agg({
            'text': lambda x: x.iloc[0].replace("\n", "</br>"),
            'golden_signal': ['first', 'count']
        }).reset_index()

        df_for_summary_html.columns = ['d_tid', 'file_names', 'text', 'gs', 'd_tid_count']

        total_log_lines = len(df_inference_csv)
        
        df_for_summary_html["coverage"] = df_for_summary_html["d_tid_count"].apply(lambda count: (count / total_log_lines) * 100)

        print(f"Number of tids with GS: {len(df_for_summary_html)}")

        # Sorting by information and count
        df_for_summary_html["information"] = df_for_summary_html["gs"].apply(lambda x: 1 if x == "Info" else 0)
        df_for_summary_html = df_for_summary_html.sort_values(by=['information', 'd_tid_count'], ascending=[True, True])

        print("Data Frame Summary is as follows=>")
        print(df_for_summary_html.head())

        # Dropping the 'text' column if present
        if not df_inference_csv.empty:
            df_inference_csv.drop(columns=['text'], inplace=True)
        if not df_inference_csv_only_non_info.empty:
            df_inference_csv_only_non_info.drop(columns=['text'], inplace=True)

        # Grouping by 'epoch' to create anomaly DataFrames
        df_inference_csv['group'] = df_inference_csv['epoch'] // 30
        df_for_anomaly_html = df_inference_csv.groupby('group').agg({
            'test_ids': ' '.join,
            'error_test_ids': ' '.join,
            'file_names': '\n'.join,
            'text_output': '\n'.join,
            'epoch': 'first',
            'golden_signal': ' '.join
        }).reset_index()

        del df_inference_csv

        df_inference_csv_only_non_info['group'] = df_inference_csv_only_non_info['epoch'] // 30
        df_for_anomaly_html_non_info = df_inference_csv_only_non_info.groupby('group').agg({
            'test_ids': ' '.join,
            'error_test_ids': ' '.join,
            'file_names': '\n'.join,
            'text_output': '\n'.join,
            'epoch': 'first',
            'golden_signal': " ".join
        }).reset_index()

        del df_inference_csv_only_non_info
        gc.collect()

        # Adding additional columns for the anomaly DataFrames
        df_for_anomaly_html['all_info'] = df_for_anomaly_html['golden_signal'].apply(lambda gs: all(item.strip() == "Info" for item in gs.split()))
        df_for_anomaly_html['len'] = df_for_anomaly_html['test_ids'].apply(lambda seq: len(seq.split()))

        if not df_for_anomaly_html_non_info.empty:
            df_for_anomaly_html_non_info['len'] = df_for_anomaly_html_non_info['test_ids'].apply(lambda seq: len(seq.split()))
        else:
            df_for_anomaly_html_non_info['len'] = []

        print(f"Total Windows: {len(df_for_anomaly_html)}")
        
        return df_for_anomaly_html, df_for_summary_html, temp_id_to_signal_map, temp_id_to_rep_log, df_for_anomaly_html_non_info
