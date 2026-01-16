import os
import pandas as pd
from tqdm import tqdm
from pandarallel import pandarallel
import argparse
from dateutil import parser as god_parse
import json
import csv
from pathlib import Path

pandarallel.initialize(progress_bar=False)
tqdm.pandas()

def get_start_end(start_str, duration):
    parsed_date = god_parse.parse(start_str)
    epoch = parsed_date.timestamp()
    return epoch, epoch+duration
    
def get_df(path, id_name):
    files = list(Path(path).glob('*.json'))
    dataframes = []
    
    for file in files:
        with file.open(mode='r') as reader:
            d = json.load(reader) 
        logs = d.keys()
        ids = d.values()
        
        df = pd.DataFrame.from_dict({"truncated_text": logs, f"{id_name}": ids})
        dataframes.append(df)
    
    if len(dataframes) == 0:
        dataframes = [pd.DataFrame(columns=['truncated_text', id_name])]
    
    return pd.concat(dataframes, ignore_index=True).drop_duplicates()

if __name__ == "__main__":

    # Define command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file_test', type=str, required=True, help='path_to_input_files')
    parser.add_argument('--input_file_test_csv', type=str, required=True, help='path_to_input_files')
    parser.add_argument('--output_dir', type=str, required=True, help='path to dir where file will be saved')
    parser.add_argument('--output_file_name', type=str, required=True, help='path to dir where file will be saved')
    parser.add_argument('--start_time', type=str, default="none", help="human readable start time string")
    parser.add_argument('--duration',type=str, default="none", help="duration of input required in seconds")
    parser.add_argument('--anomaly_flag',type=str, default="false", help="anomaly flag")
    parser.add_argument('--input_file_train',type=str, default="false", help="tm train matcher output")
    args = parser.parse_args()

    if args.start_time != "none" and args.duration == "none":
        print("durtion can't be none when start_time is not none")
        exit(101)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Directory '{args.output_dir}' created.")
    else:
        print(f"Directory '{args.output_dir}' already exists.")
    
    df_test_ids = get_df(args.input_file_test, "test_ids")
    print(df_test_ids)
    if len(df_test_ids) == 0:
        for col in df_test_ids:
            df_test_ids[col] = df_test_ids[col].astype(object)
    
    df_preprocessed_file = pd.read_csv(args.input_file_test_csv, quoting=csv.QUOTE_NONE, quotechar='',escapechar='\\')
    print(df_preprocessed_file)

    df_merged = pd.merge(df_test_ids, df_preprocessed_file, on='truncated_text', how='inner')
    
    if args.anomaly_flag == "true":
        df_train_ids = get_df(args.input_file_train, "train_ids")
        df_merged = pd.merge(df_merged, df_train_ids, on='truncated_text', how='inner')
        
    df_merged["format"] = ["%Y-%m-%d %H:%M:%S"]*len(df_merged)
    
    print(df_merged)

    df_merged = df_merged.dropna()
    df_merged = df_merged.sort_values(by='epoch')

    if args.start_time == "none":
        df_merged.to_csv(f"{args.output_dir}/{args.output_file_name}.csv", index=False, quoting=csv.QUOTE_NONE, quotechar='',escapechar='\\')
    else:
        start, end = get_start_end(args.start_time, int(args.duration))
        df_merged = df_merged[(df_merged['epoch'] >= start) & (df_merged['epoch'] <= end)]
        df_merged.to_csv(f"{args.output_dir}/{args.output_file_name}.csv", index=False, quoting=csv.QUOTE_NONE, quotechar='',escapechar='\\')
