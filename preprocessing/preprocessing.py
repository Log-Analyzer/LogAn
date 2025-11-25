import re
import configparser
import json
import glob
import ast
import os
import pytz
import math
import pandas as pd
import csv

from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timedelta
import patoolib
from pandarallel import pandarallel
import time
import numpy as np
from dateutil import parser as god_parse

from preprocessing import file_utils, pyrbras

# Global variables
rbr = None
is_initialized = False

# Constants
DEFAULT_CONFIG = "config.ini"
DEFAULT_CONFIG_SECTION = "preprocessing"

# Initialize pandarallel
pandarallel.initialize(progress_bar=True)

def initialize_once():
    """
    Initializes the rbr model if they haven't been initialized already.
    This function will be called inside each worker.
    """
    global rbr, timezone_dict, is_initialized
    if not is_initialized:
        rbr = pyrbras.load_model(os.path.join(os.path.dirname(__file__), 'model', 'manifest.json'))
        is_initialized = True

class Preprocessing:
    def __init__(self, debug_mode):
        self.df = None

        self.debug_mode = debug_mode
        self.pattern_txt = re.compile(r'^.+\.txt(\.-)?\d*$')
        self.pattern_log = re.compile(r'^.+\.log(\.-)?\d*$')

        self.ip_pattern = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.non_alphanumeric_pattern = re.compile(r'[^a-zA-Z0-9]')
        self.digit_pattern = re.compile(r'\d+')
        self.split_pattern = re.compile(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])')
        self.whitespace_pattern = re.compile(r'\s+')
        self.continuos_spaces = re.compile(r'[^\S\n]+')
        self.z_threshold = 2
        self.master_timestamp_list, self.master_format_list = self.get_master_lists()

        # Load timezones from file
        timezones_file = os.path.join(os.path.dirname(__file__), 'timezones.json')
        self.timezone_dict = json.load(open(timezones_file, 'r'))

        config = configparser.ConfigParser()
        config_file = os.path.join(os.path.dirname(__file__), DEFAULT_CONFIG)
        config.read(config_file)

        self.is_json_present = config[DEFAULT_CONFIG_SECTION]['is_json_present'] 
        self.json_message_field = ast.literal_eval(config.get(DEFAULT_CONFIG_SECTION, 'json_message_field')) if self.is_json_present == "true" else []
        self.json_time_field = ast.literal_eval(config.get(DEFAULT_CONFIG_SECTION, 'json_time_field')) if self.is_json_present == "true" else []
        
        self.is_csv_present = config[DEFAULT_CONFIG_SECTION]['is_csv_present']
        self.csv_message_field = ast.literal_eval(config.get(DEFAULT_CONFIG_SECTION, 'csv_message_field')) if self.is_csv_present == "true" else []
        self.csv_time_field = config.get(DEFAULT_CONFIG_SECTION, 'csv_time_field') if self.is_csv_present == "true" else []
            
    def get_master_lists(self):
        """
        Reads configuration files and compiles lists of timestamp formats and patterns.
        
        Returns:
            master_timestamp_list (list): Compiled list of timestamp regex patterns.
            master_format_list (list): Compiled list of corresponding timestamp formats.
        """

        config_files = glob.glob(os.path.join(os.path.dirname(__file__), DEFAULT_CONFIG))

        master_timestamp_list, master_format_list = [], []
        for config_file in config_files:
            config = configparser.ConfigParser()

            config.read(config_file)

            timestamp_formats = config.get(DEFAULT_CONFIG_SECTION, "formats")
            timestamp_formats = ast.literal_eval(timestamp_formats)
            
            timestamp_patterns = config.get(DEFAULT_CONFIG_SECTION, 'timestamps')
            timestamp_patterns = ast.literal_eval(timestamp_patterns)
            timestamp_patterns = [re.compile(pattern) for pattern in timestamp_patterns]

            master_timestamp_list.extend(timestamp_patterns)
            master_format_list.extend(timestamp_formats)

        master_timestamp_list.append(re.compile(r'[0-9A-Fa-f]{8}(\\.[0-9A-Fa-f]{4})?'))
        master_format_list.append("hexadecimal")

        # Sort master_timestamp_list based on the descending order of length
        sorted_indices = sorted(range(len(master_timestamp_list)), key=lambda x: len(master_timestamp_list[x].pattern), reverse=True)
        master_timestamp_list = [master_timestamp_list[i] for i in sorted_indices]

        # Reorder master_format_list according to the sorted indices
        master_format_list = [master_format_list[i] for i in sorted_indices]

        return master_timestamp_list, master_format_list

    def count_alphabets_and_digits(self, logline):
        """
        counts the number of occurences of alphabets and numbers in a given logline
        
        Args:
            logline (str): log.
        Returns:
            alphabet-count (int)
            digit-count (int)
        """
        alphabet_count = sum(c.isalpha() for c in logline)
        digit_count = sum(c.isdigit() for c in logline)
        return alphabet_count, digit_count

    def preprocess_logs(self, logs):
        
        # Preprocessing steps of logline for gs and fc predictions
        logs = self.ip_pattern.sub("IPADDRESS", logs)
        logs = self.url_pattern.sub("URL", logs)
        logs = self.non_alphanumeric_pattern.sub(' ', logs)
        logs = self.digit_pattern.sub('', logs)
        logs = self.split_pattern.sub(' ', logs)
        logs = self.whitespace_pattern.sub(' ', logs)
        logs = logs.lower().strip()
        return logs

    def compute_preprocessing_statistics(self, all_files, num_log_lines_processed, time, output_dir):
        """
        computes statistics of preprocessing code
        
        Args:
            all_files (list): list of files processed
            num_log_lines (int): number of logs processed.
            time (int): time took to performe preprocessing.
            output_dir (str): serialized json object as a string.
        Returns:
            None
        """
        total_size_bytes = 0
        num_log_lines_total = 0
        num_log_lines_whitespaces = 0

        # Calculate file statistics
        for fp in all_files:
            total_size_bytes += os.path.getsize(fp)
            num_log_lines_total += file_utils.count_file_lines(fp)
            num_log_lines_whitespaces += file_utils.count_file_line_whitespaces(fp)

        metrics = {
            'file_size_bytes': total_size_bytes,
            'num_log_lines_processed': num_log_lines_processed,
            "num_log_lines_total": num_log_lines_total,
            "num_log_lines_whitespaces": num_log_lines_whitespaces,
            'preprocessing_time_ms': time
        }

        with open(os.path.join(output_dir, "metrics", "preprocessing.json"), 'w') as writer:
            writer.write(json.dumps(metrics, indent=4))

    def get_time_delta(self, time_range):
        """
        computes time delta for filtering the data
        
        Args:
            time_range (str): duration of data to process
        Returns:
            (int): time delta in seconds
        """
        day = 24 * 60 * 60
        time_deltas = {
            "1-day": 1 * day,
            "2-day": 2 * day,
            "3-day": 3 * day,
            "4-day": 4 * day,
            "5-day": 5 * day,
            "6-day": 6 * day,
            "1-week": 7 * day,
            "2-week": 14 * day,
            "3-week": 21 * day,
            "1-month": 30 * day
        }
        return time_deltas.get(time_range, 7 * day)

    def is_valid_json_object(self, json_object):
        """
        checks if the input string is a json object
        
        Args:
            json_obj (str): serialized json object as a string.
        Returns:
            (bool)
        """
        if isinstance(json_object, dict) and len(json_object) > 0:
            return True
        else:
            return False

    def detect_jsons(self, logs):
        """
        For a given list of logs, this fucntion finds the logs which are json objects or multiline logs
        
        Args:
            logs (list): list of logs
        Returns:
            multiline_logs (list): list of multiline logs
            json_logs (list): list of json objects
        """
        multiline_logs, json_logs = [], []

        for log in logs:
            try:
                json_object = json.loads(log)
                if self.is_valid_json_object(json_object):
                    json_logs.append(json_object)
                else:
                    multiline_logs.append(log)
            except Exception as e:
                multiline_logs.append(log)

        return multiline_logs, json_logs

    def process_files(self, file_list):
        """
        For a given list of file, this fucntion outputs two dataframe one for multiline logs and another for json objects
        
        Args:
            file_list (list): list of input file paths
        Returns:
            multiline_df (dataframe): two columns dataframe containing logs and corresponding filenames
            json_df (dataframe): two columns dataframe containing json objects and corresponding filenames
        """
        logs = []
        json_logs_list = []
        file_names_multiline = []
        file_names_json = []
        
        for file in tqdm(file_list):
            with open(file, "r", encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                multiline_logs, json_logs = self.detect_jsons(lines)

                logs.extend(multiline_logs)
                json_logs_list.extend(json_logs)

                file_names_multiline.extend([file] * len(multiline_logs))
                file_names_json.extend([file] * len(json_logs))

        multiline_df = pd.DataFrame({"text": logs, "file_names": file_names_multiline})
        json_df = pd.DataFrame({"text": json_logs_list, "file_names": file_names_json})
        return multiline_df, json_df
    
    def flatten_json(self, json_object, sep="_"):
        """
        For a given nested json, this function flattens the json object with "_" as a connected between keys
        
        Args:
            json_object (json): input json object
        Returns:
            result (json): flattened version of input json
        """
        stack = [((), json_object)]
        result = {}
        while stack:
            parent_keys, node = stack.pop()
            for k, v in node.items():
                new_key = parent_keys + (k,)
                if isinstance(v, dict):
                    stack.append((new_key, v))
                else:
                    result[sep.join(new_key)] = v
        return result

    def is_string_numeric(self, NumberString):
        """
        For a given string, this function checks if the string represents a number (int or float)
        
        Args:
            NumberString (str): numeric string
        Returns:
            Number (int/float): numerical version of input
            Flag (Bool): True if number is int or float else False
        """
        if isinstance(NumberString, int) or isinstance(NumberString, float):
            return None, False
        
        elif NumberString.isdigit():
            Number = int(NumberString)
            return Number, True
        else:
            try:
                Number = float(NumberString)
                return Number, True
            except:
                return None, False

    def timestamp_json_csv_to_epoch(self, time_stamp_str):
        """
        Args:
            time_stamp_str (str): time stamp string extracted from json object or csv column
        Returns:
            time_stamp_str (string): time stamp string extracted from json object or csv column
            Epoch (int): epoch of the time stamp string extracted from json object or csv column
        """

        number, numeric_flag = self.is_string_numeric(time_stamp_str)
        current_epoch = datetime.now().timestamp()
        if isinstance(time_stamp_str, int) or isinstance(time_stamp_str, float):
            if time_stamp_str > current_epoch:
                time_stamp_str = time_stamp_str/1000
                
            return time_stamp_str, time_stamp_str
        
        elif numeric_flag:
            time_stamp_str = number
            if time_stamp_str > current_epoch:
                time_stamp_str = time_stamp_str/1000
                
            return time_stamp_str, time_stamp_str
        
        else:
            try:
                ts = god_parse.parse(time_stamp_str).timestamp()
            except Exception as e:
                return None, None
            else:
                return time_stamp_str, ts
    
    def process_fn_json(self, json_obj):
        """
        This function processed one json object at a time: extracts the log and timestamp out of a json object
        Args:
            json_obj (json): input json object
        Returns:
            timestamp (string): time stamp string extracted from json the object
            ts (int): epoch of the time stamp string extracted from json object or csv column
            log (str): logline extracted from json object
            preprocessed_text(str): preprocessed version of the logline extracted from json object
            digit_count(int): number of digits present in the logline extracted from json object
            alphabet_count + digit_count + 1(int): 1{it is added for +1 smoothing} + total characters (digit+alphabet_count) present in the logline extracted from json object
            len(log.split(" ")) (int): number of tokens in the logline extracted from json object
            Flag (bool): it's True when json is not discarded else it is False
        """
        check_nested_strings = lambda strings: any(len(string.split('_')) > 1 for string in strings)
        flatten_flag = check_nested_strings(self.json_time_field) or check_nested_strings(self.json_message_field)
        if flatten_flag:
            json_object = self.flatten_json(json_obj)
        else:
            json_object = json_obj

        if len(self.json_time_field) > 0:
            json_time_field_temp = set(self.json_time_field) & set(json_object.keys())
            if len(json_time_field_temp) > 0:
                json_time_field_temp = json_time_field_temp.pop()
            else:
                return None, None, json_obj, None, None, None, None, True
        else:
            return None, None, json_obj, None, None, None, None, True

        if len(self.json_message_field) > 0:
            json_message_field_temp = set(self.json_message_field) & set(json_object.keys())

            if len(json_message_field_temp) > 0:
                json_message_field_temp = json_message_field_temp.pop()
            else:
                return None, None, json_obj, None, None, None, None, True
        else:
            return None, None, json_obj, None, None, None, None, True
        
        time_stamp_str, log =  json_object[json_time_field_temp], json_object[json_message_field_temp].strip()
        
        timestamp, ts = self.timestamp_json_csv_to_epoch(time_stamp_str)
        
        if ts:
            log = log.encode('ascii', 'ignore').decode('ascii')
            log = self.continuos_spaces.sub(' ', log)
            
            preprocessed_text = self.preprocess_logs(log)

            log_without_ts = log.replace(timestamp, "") if timestamp else log
            alphabet_count, digit_count = self.count_alphabets_and_digits(log_without_ts)
                
            return timestamp, ts, log, preprocessed_text, digit_count, alphabet_count + digit_count + 1, len(log.split(" ")), False
        
        return None, None, json_obj, None, None, None, None, True

    def hex_to_timestamp(self, hex_string):
        """
        Converts a hexadecimal string to a timestamp.
        
        Args:
            hex_string (str): Hexadecimal string representing a timestamp.
        
        Returns:
            datetime_object (datetime or None): Converted datetime object, None if invalid.
            future_flag (bool): Indicates if the datetime is in the future.
        
        Example:
            hex_to_timestamp('5f2c6e00')  # Example hexadecimal timestamp
        """

        current_datetime = datetime.now()
        current_year = current_datetime.year
        # Convert hexadecimal string to decimal
        decimal_timestamp = int(hex_string.split(".")[0], 16)
        datetime_object = datetime.fromtimestamp(decimal_timestamp)
        future_flag = False

        if datetime_object.year < current_year-1:
            return None, future_flag
        
        if datetime_object > current_datetime:
            future_flag = True
            return datetime_object, future_flag

        return datetime_object, future_flag

    def epoch_str_to_timestamp(self, epoch_str):

        """
        Converts an epoch string to a datetime object.
        
        Args:
            epoch_str (str): Epoch timestamp string.
        
        Returns:
            datetime_object (datetime or None): Converted datetime object, None if invalid.
            future_flag (bool): Indicates if the datetime is in the future.
        
        Example:
            epoch_str_to_timestamp('1596230000')  # Example epoch timestamp
        """

        current_datetime = datetime.now()
        current_year = current_datetime.year
        # Convert hexadecimal string to decimal
        decimal_timestamp = int(epoch_str)

        if current_datetime.timestamp() < decimal_timestamp:
            decimal_timestamp = decimal_timestamp / 1000

        datetime_object = datetime.fromtimestamp(decimal_timestamp)

        future_flag = False

        if datetime_object.year < current_year-1:
            return None, future_flag
        
        if datetime_object > current_datetime:
            future_flag = True
            return datetime_object, future_flag

        return datetime_object, future_flag

    def day_of_the_year(self, match):

        """
        Custom function to parse dates like: 24165 16:48:54.18 or 2024165 16:48:54.18
        interpretation: 16:48:54.18 of 165th day of year 2024 
        
        Returns:
            datetime_object (datetime or None): Converted datetime object, None if invalid.
            future_flag (bool): Indicates if the datetime is in the future.
        """

        year_suffix = int(match.group(1))
        year = 2000 + year_suffix  # Convert to 4-digit year
        day_of_year = int(match.group(2))
        time_str = match.group(3)

        date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)

        datetime_str = date.strftime("%Y-%m-%d") + " " + time_str

        datetime_format = "%Y-%m-%d %H:%M:%S.%f"

        datetime_object = datetime.strptime(datetime_str, datetime_format)

        future_flag = False
        current_datetime = datetime.now()
        if datetime_object > current_datetime:
            future_flag = True
            
        return datetime_object, future_flag

    def master_datetime_extractor(self, log, timezone_dict, master_timestamp_list, master_format_list):
        """
        Extracts datetime from log using multiple patterns and formats.
        
        Args:
            log (str): Log text to extract datetime from.
            timezone_dict (dict): Dictionary of timezone information.
            master_timestamp_list (list): list of compiled master regexes
            master_format_list (list): list of master time formats

        Returns:
            match (str or None): Extracted timestamp as a string, None if not found.
            timestamp (float or None): Extracted timestamp in epoch format, None if not found.
            future_flag (bool): Indicates if the parsed date is in the future.
        """

        future_flag = False
        for pattern, format in zip(master_timestamp_list, master_format_list):
            match_original = re.search(pattern, log)
            if match_original:
                start = match_original.start()
                end = match_original.end()
                match = match_original.group()
                match_output = match
                # Checks
                if not (
                    (start > 0 and log[start - 1] == '[' and end < len(log)) or # If timestamp is enclosed in square brackets [], eg '[2021-07-01 12:00:00]'
                    (format == "hexadecimal" and start > 0 and log[start - 1] == '(' and end < len(log)) or # If timestamp is enclosed in parentheses (), eg '(5f2c6e00)'
                    (format == "hexadecimal" and start > 0 and log[start - 1] == '+' and end < len(log)) or # If timestamp is prefixed with a plus sign, eg '+5f2c6e00'
                    (start > 0 and log[start - 1].isspace()) or # If timestamp is prefixed with a space eg ' 2021-07-01 12:00:00'
                    (start == 0) or # If timestamp is at the beginning of the log eg '2021-07-01 12:00:00 log message'
                    (start > 0 and log[start - 1] == '=') or # If timestamp is prefixed with an equals sign eg 'time=2021-07-01 12:00:00'
                    (start > 0 and log[start - 1] == '"' and end < len(log) and log[end] == '"')  # If timestamp is enclosed in double quotes eg '"2021-07-01 12:00:00"'
                 ):
                    continue  
                
                if format == "hexadecimal":
                    if start > 5:
                        continue
                    else:
                        dex_date_obj, future_flag = self.hex_to_timestamp(match)
                        if dex_date_obj == None:
                            continue
                        return match_output, dex_date_obj.timestamp(), future_flag
                    
                elif format == "epoch":
                    if start > 5:
                        continue
                    else:
                        dex_date_obj, future_flag = self.epoch_str_to_timestamp(match)
                        if dex_date_obj == None:
                            continue
                        return match_output, dex_date_obj.timestamp(), future_flag  
                
                elif format == "day_of_the_year":
                    date_obj, future_flag = self.day_of_the_year(match_original)
                    return match_output, date_obj.timestamp(), future_flag

                try:
                    parsed_date = god_parse.parse(match, tzinfos=timezone_dict)
                except:
                    found_zone = False
                    for z, timezone in timezone_dict.items():
                        if z in match:
                            match = match.replace(z, "").strip()
                            tz = pytz.timezone(timezone)
                            found_zone = True
                            break
                    
                    match = re.sub(r"[A-Za-z]+$", "", match).strip()                  
                    try:
                        parsed_date = datetime.strptime(match, format)
                        if found_zone:
                            parsed_date = tz.localize(parsed_date)
                        else:
                            gmt = pytz.timezone('GMT')
                            parsed_date = parsed_date.astimezone(gmt)
                    except Exception:
                        continue
                if parsed_date is None:
                    return match_output, None, False
                
                current_datetime = datetime.now()
                future_flag = parsed_date.timestamp() >= current_datetime.timestamp()
                return match_output, parsed_date.timestamp(), future_flag
    
        return None, None, future_flag

    def aql_datetime_extractor(self, log, rbr, timezone_dict):
        """
        This function outputs the timestamp, epoch present in the input logline
        Args:
            log (str): one line of multiline log
            rbr (aql object): Thanks to aql team for creating a tool which extracts out lots of unknown unkown timestamps
            timezone_dict (dict): dictionary containing timezone info
        Returns:
            timestamp (string): time stamp string extracted from the line. It can be None if no timestamp is present in the line
            ts (int): epoch of the time stamp string extracted the line or csv column
            future_flag (Bool): True if parsed date is in future else False
        """
        current_datetime = datetime.now()
        current_day = current_datetime.day
        result = json.loads(rbr.process(log, "en"))
        future_flag = False
        try:
            match = result['annotations']['DateTimeOutput'][0]['span']['text']
            parsed_date = god_parse.parse(match, tzinfos=timezone_dict)

            if parsed_date.hour == 0 and parsed_date.minute == 0 and parsed_date.second == 0:
                return None, None, future_flag
            
            if parsed_date.timestamp() > current_datetime.timestamp():
                parsed_date = parsed_date.replace(day=current_day-1)

            if parsed_date.timestamp() > current_datetime.timestamp():
                future_flag = True
                return match, parsed_date.timestamp(), future_flag

            return match, parsed_date.timestamp(), future_flag
        except:
            return None, None, future_flag

    def extract_ts(self, log, rbr, timezone_dict, master_timestamp_list, master_format_list):
        """
        This function outputs the timestamp, epoch present in the input logline
        Args:
            log (str): one line of multiline log
            rbr (aql object): Thanks to aql team for creating a tool which extracts out lots of unknown unkown timestamps
            timezone_dict (dict): dictionary containing timezone info
            master_timestamp_list (list): list of compiled master regexes
            master_format_list (list): list of master time formats
        Returns:
            timestamp (string): time stamp string extracted from the line. It can be None if no timestamp is present in the line
            ts (int): epoch of the time stamp string extracted the line or csv column
        """
        ts = None
        future_flag = False
        timestamp, ts, future_flag = self.master_datetime_extractor(log, timezone_dict, master_timestamp_list, master_format_list)
        if not ts:
            timestamp, ts, future_flag = self.aql_datetime_extractor(log, rbr, timezone_dict)
        
        if not future_flag:
            return timestamp, ts
        
        return None, None

    def process_fn(self, log, timezone_dict, master_timestamp_list, master_format_list):
        """
        This function processes one line of multiline log at a time:  extracts timestamp present in the logline if possible
        Args:
            log (str): one line of multiline log
            timezone_dict (dict): dictionary containing timezone info
            master_timestamp_list (list): list of compiled master regexes
            master_format_list (list): list of master time formats
        Returns:
            timestamp (string): time stamp string extracted from the line. It can be None if no timestamp is present in the line
            ts (int): epoch of the time stamp string extracted the line or csv column
            log (str): logline extracted the line
            preprocessed_text(str): preprocessed version of the logline extracted the line
            digit_count(int): number of digits present in the logline extracted the line
            alphabet_count + digit_count + 1(int): 1{it is added for +1 smoothing} + total characters (digit+alphabet_count) present in the logline extracted the line
            len(log.split(" ")) (int): number of tokens in the logline extracted the line
        """
        # Ensure initialization only happens once per worker
        initialize_once()

        timestamp, ts = self.extract_ts(log, rbr, timezone_dict, master_timestamp_list, master_format_list)
        log = log.encode('ascii', 'ignore').decode('ascii')
        log = self.continuos_spaces.sub(' ', log)
        
        preprocessed_text = self.preprocess_logs(log)

        log_without_ts = log.replace(timestamp, "") if timestamp else log
        alphabet_count, digit_count = self.count_alphabets_and_digits(log_without_ts)
            
        return timestamp, ts, log, preprocessed_text, digit_count, alphabet_count + digit_count + 1, len(log.split(" "))

    def preprocess(self, input_files, time_range, output_dir, process_log_files, process_txt_files):
        """
        Preprocess a set of log files by filtering, processing, and performing truncation of logs using statistical analysis on the logs length.
        
        This function takes a list of input files or directories, processes log data from the files (or from within
        json objects), filters out unnecessary files, processes the relevant log files.

        Args:
            input_files (list): List of file paths or directories to process. Each file can be a log file or a directory 
                                containing log files.
            time_range (str): Time range filter for the logs. Can be a specific range or "all-data" to include everything.
            output_dir (str): Directory path where processed files, ignored files, and plots will be saved.
            process_log_files (boolean): Flag to process LOG files from folder. This will not affect files that are explicitly provided by the user.
            process_txt_files (boolean): Flag to process TXT files from folder. This will not affect files that are explicitly provided by the user.

        Steps:
            1. Log file filtering:
                - Identify files that are archives, directories, or log files (.csv, .xml, .tsv, .xlsx).
                - Extract relevant log files from directories, while ignoring others.
                - Append non-log files to an ignored list for logging purposes.
            
            2. Write ignored and processed file lists:
                - Save ignored files and processed files lists into the output directory.
            
            3. Process log files:
                - Creates two dataframes by reading all the input files. One df is for json objects found in the file and another one is for multiline logs
                - process json dataframe in parallel to extract timestamp from logline.
                - process multiline dataframe in parallel to extract timestamp from logline.
            
            4. Preprocess and aggregate logs:
                - Group logs based on their file names and timestamps.
                - Aggregate the logs by combining text and preprocessed text entries, summing numeric values, 
                and keeping track of timestamps and epoch values.
                - Merge processed JSON and log data together.
            
            5. Apply time filtering:
                - Filter the logs based on the provided time range (if specified).

            6. Statistical analysis:
                - Calculate the mean and standard deviation of token counts in the logs.
                - Apply Z-score truncation to handle logs with excessively high token counts.
                - Generate and save histograms for the original and truncated token counts.
            
            7. Save the results:
                - The final DataFrame containing preprocessed logs is stored in `self.df`.
                - The preprocessing statistics are computed and saved.
            
        Logging and Output:
            - The function logs relevant processing steps and intermediate results.
            - Outputs two files: `ignored_files.log` and `processed_files.log`, along with a histogram plot (`fig.png`).
            - The processed logs are stored in the DataFrame `self.df` and statistical summaries are computed.

        Example usage:
            preprocess(input_files=["log1.txt", "log_directory"], time_range="last-24-hours", output_dir="/path/to/output")

        Notes:
            - The function uses parallel processing for text and JSON object processing.
            - Files that do not match the expected patterns or are archived are ignored.
            - It truncates the logs where the token count is unusually high using Z-score truncation.
        
        """
        start_time = time.time()  # Start the timer to track the duration of the process

        print("Input files:")
        print(input_files)

        files_to_process = []  # List to store files that will be processed
        ignored_list = []  # List to store ignored files (e.g., archives)

        # Iterate through each file in input_files
        for file_ in input_files:

            # Print file information and check if it's an archive or directory
            print(file_, f"patoolib.is_archive(file_): {patoolib.is_archive(file_)}")
            print(file_, f"os.path.isdir(file_): {os.path.isdir(file_)}")

            extensions = Path(file_).suffixes  # Get file extensions
            print(f"'.csv' in extensions: {'.csv' in extensions}")
            print(f"'.xml' in extensions: {'.xml' in extensions}")

            # Skip archives or irrelevant file types
            if patoolib.is_archive(file_):
                ignored_list.append(file_)

            # Handle directories by finding log files
            elif os.path.isdir(file_):
                all_files_in_dir = [fp for fp in glob.glob(os.path.join(file_, '**'), recursive=True) if not os.path.isdir(fp)]

                log_files, txt_files = [], []
                if process_txt_files:
                    txt_files = [file for file in all_files_in_dir if self.pattern_txt.match(os.path.basename(file))]
                if process_log_files:
                    log_files = [file for file in all_files_in_dir if self.pattern_log.match(os.path.basename(file))]

                files_to_process.extend(log_files + txt_files)
                ignored_list.extend([fp for fp in all_files_in_dir if fp not in (log_files + txt_files)])

            # Handle individual files based on their extensions
            elif ('.xml' in extensions) or (self.is_csv_present == "false" and ('.csv' in extensions or '.xlsx' in extensions)) or ('.tsv' in extensions):
                ignored_list.append(file_)
            else:
                files_to_process.append(file_)

        # Remove directories from the list of files to process and remove duplicates
        files_to_process = list(set(fp for fp in files_to_process if not Path(fp).is_dir()))

        # Log files to be processed
        print("Files to process:")

        # Write ignored and processed files to log files for debugging purposes
        with open(os.path.join(output_dir, "developer_debug_files", "ignored_files.log"), 'w') as writer:
            writer.write("\n".join(ignored_list))

        with open(os.path.join(output_dir, "developer_debug_files", "processed_files.log"), 'w') as writer:
            writer.write("\n".join(files_to_process))
        
        print(*files_to_process, sep="\n")

        # Process the log files and get dataframes for logs and JSON objects
        df, df_json = self.process_files(files_to_process)

        # Process JSON data in parallel
        df_json = df_json.dropna()
        if len(df_json) > 0:
            df_json[['timestamps', 'epoch', 'text', 'preprocessed_text', 'numeric_count', "total_count", "token_count", "discarded"]] = df_json['text'].parallel_apply(lambda json_obj: pd.Series(self.process_fn_json(json_obj)))
        else:
            df_json = pd.DataFrame({
                'timestamps': [],
                'epoch': [],
                'text': [],
                'preprocessed_text': [],
                'numeric_count': [],
                'total_count': [],
                'token_count': [],
                'discarded': []
            })

        if (self.debug_mode == "true"):
            df_json_discarded = df_json[df_json["discarded"] == True]
            df_json_discarded = df_json_discarded['text'] # Extract discarded JSON objects
            with open(os.path.join(output_dir, "developer_debug_files", "json_discarded.log"), 'w') as log_file:
                for line in df_json_discarded:
                    log_file.write(f"{line}\n")  # Write each line of the 'text' column to the log file
        df_json = df_json[df_json["discarded"] == False]
        df_json = df_json.drop(columns=["discarded"])

        print(f"Debug mode is set to: {self.debug_mode}")
        # Process multiline log data in parallel
        print("Starting pandarallel for log processing")
        if len(df) > 0:
            df[['timestamps', 'epoch', 'text', 'preprocessed_text', 'numeric_count', "total_count", "token_count"]] = df['text'].parallel_apply(lambda log: pd.Series(self.process_fn(log, self.timezone_dict, self.master_timestamp_list, self.master_format_list)))
        else:
            df = pd.DataFrame({
                'file_names': [],
                'timestamps': [],
                'epoch': [],
                'text': [],
                'preprocessed_text': [],
                'numeric_count': [],
                'total_count': [],
                'token_count': [],
            })

        if (self.debug_mode == "true"):
            df_none_logs = df[df["epoch"].isna()]  # Filter rows where 'epoch' is None (or NaN)
            text_column = df_none_logs['text']  # Extract the 'text' column
            # Write the 'text' column to the .log file
            with open(os.path.join(output_dir, "developer_debug_files", "none_logs.log"), 'w') as log_file:
                for line in text_column:
                    log_file.write(f"{line}\n")  # Write each line of the 'text' column to the log file
        
        # Filter out logs of files which doesn't contains any timestamps. THE SPECIAL CASE :)!
        df_groups_all_none = df.groupby("file_names").filter(lambda x: x['epoch'].isna().all())

        # Group logs by file names and epoch timestamps
        df["group"] = df.groupby("file_names")["epoch"].transform(lambda x: (~x.isna()).cumsum())

        # Keep only valid logs (where epoch is not None)
        df = df.loc[~df.index.isin(df_groups_all_none.index)]

        # Aggregate logs based on file names and group number
        df = df.groupby(["file_names", "group"]).agg({
            "text": "\n".join,
            "preprocessed_text": "\n".join,
            "epoch": "first",
            "timestamps": "first",
            "numeric_count": "sum",
            "total_count": "sum",
            "token_count": "sum"
        }).reset_index()

        df = df.drop(columns=["group"])

        # Combine processed logs and JSON data
        df = pd.concat([df, df_json], axis=0, ignore_index=True)

        if len(df) == 0:
            self.df = df
            print("PREPROCESSING ERROR: No log lines extracted from the input.")
            return

        # Calculate the ratio of numeric to total counts and filter based on threshold
        df['ratio'] = df['numeric_count'] / df['total_count']
        df = df[df["ratio"] < 0.5]
        df = df.drop(columns=["ratio"])

        # Determine the maximum epoch value and adjust timestamps
        max_epoch = df['epoch'].max()
        if math.isnan(max_epoch):
            max_epoch = datetime.now().timestamp()

        timestr = datetime.fromtimestamp(max_epoch).strftime('%Y-%m-%d %H:%M:%S')
        df_groups_all_none["epoch"] = [max_epoch]*len(df_groups_all_none)
        df_groups_all_none["timestamps"] = [timestr]*len(df_groups_all_none)

        df = pd.concat([df, df_groups_all_none], ignore_index=True)

        # Apply time range filtering if needed
        print(f"Time range provided by user: {time_range}")
        delta = self.get_time_delta(time_range)
        if time_range != "all-data":
            min_epoch = max_epoch - delta
            df = df[(df['epoch'] >= min_epoch) & (df['epoch'] <= max_epoch)]

        try:
            min_epoch = df['epoch'].min()
            min_readable = datetime.utcfromtimestamp(min_epoch).strftime('%Y-%m-%d %H:%M:%S')
        except:
            print(f"Provided data doesn't contain valid min date or the date parsing has failed!")

        max_readable = datetime.utcfromtimestamp(max_epoch).strftime('%Y-%m-%d %H:%M:%S')

        mean = df['token_count'].mean()
        std_dev = df['token_count'].std()

        if np.isnan(std_dev) or std_dev == 0:
            # No variability -> No need for z-score truncation
            df['z_score'] = 0
            df['truncated_token_count'] = df['token_count']
            df['truncated_log'] = df['text']
        else:
            z_threshold = self.z_threshold
            upper_bound = int(mean + z_threshold * std_dev)

            df['z_score'] = (df['token_count'] - mean) / std_dev
            df['truncated_token_count'] = np.where(df['z_score'] > z_threshold, upper_bound, df['token_count'])
            df['truncated_log'] = df.parallel_apply(
                lambda row: row['text'][:upper_bound] if row['token_count'] > upper_bound else row['text'], axis=1
            )


        # # Plot histograms of original and truncated token counts
        # plt.figure(figsize=(14, 7))

        # # Original token_count distribution (Log-Log Scale)
        # plt.subplot(1, 2, 1)
        # plt.hist(df['token_count'], bins=50, color='blue', alpha=0.7, label='Original')
        # plt.axvline(upper_bound, color='red', linestyle='dashed', linewidth=2, label=f'Upper Bound (Z={z_threshold})')
        # plt.title('Original Token Count Distribution (Log-Log Scale)')
        # plt.xlabel('Token Count')
        # plt.ylabel('Frequency')
        # plt.xscale('log')
        # plt.legend()

        num_truncated = (df['token_count'] != df['truncated_token_count']).sum()
        # Log the number of truncated logs
        print(f"Number of logs affected by truncation: {num_truncated}")

        # Annotate the percentage of truncated logs
        # total_logs = len(df)
        # truncated_percentage = (num_truncated / total_logs) * 100
        # plt.text(0.05, 0.95, f'Truncated Logs: {truncated_percentage:.2f}%', 
        #         transform=plt.gca().transAxes, fontsize=12, 
        #         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

        # # Truncated token_count distribution
        # plt.subplot(1, 2, 2)
        # plt.hist(df['truncated_token_count'], bins=50, color='green', alpha=0.7, label='Truncated')
        # plt.title('Truncated Token Count Distribution')
        # plt.xlabel('Token Count')
        # plt.ylabel('Frequency')
        # plt.legend()

        # plt.tight_layout()
        # plt.savefig(f"{output_dir}/fig.png")


        # Clean up the final DataFrame
        df = df.drop(columns=["total_count", "numeric_count", "token_count", "truncated_token_count", "z_score"])
        df = df.dropna()

        print("Total log lines:", len(df))
        print("Min epoch (human-readable):", min_readable)
        print("Max epoch (human-readable):", max_readable)

        # Save preprocessing statistics and final DataFrame
        self.compute_preprocessing_statistics(
            files_to_process, 
            len(df), 
            (time.time() - start_time) * 1000, 
            output_dir
        )
        self.df = df  # Store the final processed logs DataFrame in the class instance.
        if (self.debug_mode == "true"):
            df.to_csv(os.path.join(output_dir, "developer_debug_files", "processed_logs.csv"), index=False, quoting=csv.QUOTE_NONE, quotechar='',escapechar='\\')