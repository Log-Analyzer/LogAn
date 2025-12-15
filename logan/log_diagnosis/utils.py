import base64
import html
import json
import re
import os
from argparse import ArgumentParser
import shutil
import sys

from jinja2 import Environment, FileSystemLoader
import pandas as pd

from logan.telemetry.es import get_elasticsearch_config, get_feedback_index

def get_b64_encoded_credentials(username, passwd):
    credentials = f'{username}:{passwd}'
    return base64.b64encode(credentials.encode()).decode('ascii')

def row_size(row):
    """
    Calculates the memory size of a given row.
    
    Args:
        row (pd.Series): A row of the DataFrame.
    
    Returns:
        int: Size of the row in bytes.
    """
    return sys.getsizeof(row)

def replace_tags(text):
    """
    Replace specific HTML tags with &lt; and &gt; and escape HTML entities in the given text.
    Args:
        text (str): The input string containing HTML tags and entities.
    Returns:
        str: The cleaned string with specific HTML tags replaced and entities escaped.
    """
    try:
        # Replace specific tags
        text = re.sub(r'</?script>', '', text, flags=re.IGNORECASE)
        # Replace multiple consecutive <br> tags with a single <br>
        text = re.sub(r'(<br\s*/?>\s*)+', '<br>', text)
        
        # Replace all other HTML tags with &lt; and &gt;, except <br> and <br/> tags
        parts = re.split(r'(<[^>]+>)', text)
        for i in range(len(parts)):
            if parts[i].lower() not in ('<br>', '<br/>', '</br>'):
                parts[i] = html.escape(parts[i],quote=False)
        text = ''.join(parts)
    except TypeError:
        print(f"Error in replacing HTML tags for text: {text}")
    
    return text

def split_df_on_size(df, threshold):
    """
    Splits a DataFrame into chunks such that each chunk's total memory size is below the specified threshold.
    
    Args:
        df (pd.DataFrame): The input DataFrame to be split.
        threshold (int): The maximum size (in bytes) for each chunk.
    
    Returns:
        list: A list of DataFrames, each representing a chunk within the size limit.
    """
    df['row_size_bytes'] = df.apply(row_size, axis=1)  # Calculate the size of each row in bytes.

    current_size = 0  # Tracks the size of the current chunk.
    chunks = []  # Holds all the chunks created from splitting.
    chunk = []  # Holds the current chunk of rows.

    # Iterate through each row in the DataFrame.
    for index, row in df.iterrows():
        row_bytes = row["row_size_bytes"]
        # If adding the current row doesn't exceed the threshold, add it to the chunk.
        if current_size + row_bytes <= threshold:
            chunk.append(row)
            current_size += row_bytes
        else:
            # If the current chunk exceeds the threshold, store it and start a new chunk.
            if len(chunk) > 0: 
                chunks.append(pd.DataFrame(chunk))
            chunk = [row]  # Start a new chunk with the current row.
            current_size = row_bytes

    # Add the last chunk if it exists.
    if len(chunk) > 0:  
        chunks.append(pd.DataFrame(chunk))

    return chunks

def create_feedback_variable():
    """
    Creates a configuration dictionary for Elasticsearch feedback, including credentials and index information.
    
    Returns:
        dict: Configuration dictionary with Elasticsearch credentials and feedback index.
    """
    es_config = get_elasticsearch_config()  # Load Elasticsearch configuration.
    es_config['feedback_index'] = get_feedback_index()  # Set the feedback index.
    # Encode the username and password as base64 and store as a token.
    es_config['token'] = get_b64_encoded_credentials(es_config['username'], es_config['password'])  
    return es_config

def get_anomaly_html_str(df_final_anomalies, output_dir):
    """
    Generates an HTML string for anomaly data and saves the JSON representation of the data in chunks.
    
    Args:
        df_final_anomalies (pd.DataFrame): DataFrame containing anomaly information.
        output_dir (str): Directory path where output files should be saved.
    
    Returns:
        str: Rendered HTML string for the anomalies.
    """
    
    # If no anomalies are present, notify the user.
    if len(df_final_anomalies) == 0:
        print("No anomalies detected") 
    
    # Remove HTML tags from log entries.
    tag_pattern = re.compile(r'<[^>]*>')
    df_final_anomalies['list_logs'] = df_final_anomalies['list_logs'].apply(lambda log: re.sub(tag_pattern, '', log))
    
    # Split the DataFrame into smaller chunks, each under 2.5 MB.
    list_chunked_df = split_df_on_size(df_final_anomalies, threshold=2.5*1024*1024)  # 2.5 MB splits
    
    chunk_size = []  # To track the size of each chunk.
    output_prefix = f'{output_dir}/developer_debug_files/data'  # Prefix for the output JSON files.
    
    # Iterate through each chunk and save it as a JSON file.
    for idx, df_anomaly in enumerate(list_chunked_df):
        chunk_size.append(len(df_anomaly))  # Record the size of the current chunk.
        output_json_obj = []  # Prepare a list to store JSON objects.
        
        # Convert each row of the DataFrame into a JSON object.
        for _, row in df_anomaly.iterrows():
            temp_json_obj = {}
            start, end, each_window, file_window, templates = row["start_ts"], row["end_ts"], row["list_logs"], row["list_files"], row["list_templates"]

            duration = f"{start} -- \n {end}"
            logs = each_window.split('\n')  # Split logs by newline.
            files = file_window.split('\n')  # Split file names by newline.

            # Extract the last element in logs that represents the golden signal (gs).
            list_of_gs = [item.split('=>')[-1].split()[-1].strip() for item in logs]
            list_of_templates = templates.split(" ")

            temp_json_obj['duration'] = duration
            temp_json_obj['logs'] = logs
            temp_json_obj['files'] = files
            temp_json_obj['gs'] = [gs.strip() for gs in list_of_gs]
            temp_json_obj['templateIds'] = list_of_templates
            output_json_obj.append(temp_json_obj)
        
        # Save the JSON object to a file.
        output_file = f"{output_prefix}_{idx+1}.json"
        
        with open(output_file, 'w') as output_json_file:
            json.dump(output_json_obj, output_json_file, indent=4)
            
        print(f"Written chunk {idx+1} to {output_file}")
    
    # Create a Jinja environment to render the HTML template.
    path = os.path.dirname(os.path.abspath(__file__))
    env = Environment(loader=FileSystemLoader(os.path.join(path, 'templates')))
    html_template = env.get_template('anomalies.html')

    # Render the HTML template using the provided data.
    rendered_template = html_template.render(
        chunk_size=chunk_size, 
        no_of_chunks=len(list_chunked_df), 
        no_of_windows=sum(chunk_size), 
        zip=zip, set=set, 
        output_dir='../developer_debug_files', 
        min=min
    )
    
    return rendered_template

def get_summary_html_str(df_for_summary_html, include_golden_signal_dropdown, ignored_file_list, processed_file_list):
    """
    Generates an HTML string for the summary report, including details about golden signals and processed files.
    
    Args:
        df_for_summary_html (pd.DataFrame): DataFrame containing summary data.
        include_golden_signal_dropdown (bool): Whether to include a dropdown for golden signals in the report.
        ignored_file_list (list): List of files that were ignored during processing.
        processed_file_list (list): List of files that were successfully processed.
    
    Returns:
        str: Rendered HTML string for the summary report.
    """
    # Round coverage values to four decimal places.
    df_for_summary_html = df_for_summary_html[['d_tid', 'text', 'gs', 'd_tid_count', 'coverage', 'file_names']]
    df_for_summary_html['coverage'] = df_for_summary_html['coverage'].apply(lambda val: round(val, 4))
    
    df_for_summary_html['text'] = df_for_summary_html['text'].apply(replace_tags)
    
    # Create a Jinja environment to render the summary HTML template.
    path = os.path.dirname(os.path.abspath(__file__))
    env = Environment(loader=FileSystemLoader(os.path.join(path, 'templates')))
    html_template = env.get_template('summary_golden_signal_error.html')

    # Render the summary HTML template.
    rendered_template = html_template.render(
        summary_table=df_for_summary_html.values.tolist(),
        include_golden_signal_dropdown=include_golden_signal_dropdown,
        ignored_file_list=ignored_file_list,
        processed_file_list=processed_file_list,
        unique_file_names=sorted(df_for_summary_html['file_names'].unique().tolist())
    )

    return rendered_template

def prepare_output_dir(output_dir: str, clean_up: bool = False):
    """
    Prepares the output directory by creating necessary subdirectories.
    
    Args:
        output_dir (str): The directory path where output will be stored.
    """
    # Clean up output directory if it exists.
    if clean_up and os.path.exists(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)

    # Create necessary subdirectories.
    FOLDERS = ['run', 'pandarallel_cache', 'test_templates', 'log_diagnosis', 'developer_debug_files', 'metrics']
    os.makedirs(output_dir, exist_ok=True)
    for folder in FOLDERS:
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

    # Copy HTML templates libs to the output directory.
    shutil.copytree(os.path.join(os.path.dirname(__file__), 'templates', 'libs'), os.path.join(output_dir, 'log_diagnosis', 'libs'), dirs_exist_ok=True)

    # Setup Env Variables
    os.environ['MEMORY_FS_ROOT'] = os.path.join(output_dir, 'pandarallel_cache')

if __name__ == '__main__':
    """
    Main entry point for the script. Parses command-line arguments and generates either an anomaly or summary report.
    """
    parser = ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    debug_dir = os.path.join(os.path.dirname(args.output_dir), 'developer_debug_files')

    # Generate the anomaly report and save it.
    df_final_anomalies = pd.DataFrame(columns=['start_ts', 'end_ts', 'list_logs', 'list_files', 'list_templates'])    
    with open(os.path.join(args.output_dir, 'anomalies.html'), 'w') as writer:
        writer.write(get_anomaly_html_str(df_final_anomalies, output_dir=debug_dir))
    
    # Generate the summary report and save it.
    df_for_summmary_html = pd.DataFrame(columns=['d_tid', 'text', 'gs', 'd_tid_count', 'coverage', 'file_names'])
    with open(os.path.join(args.output_dir, 'summary.html'), 'w') as writer:
        writer.write(get_summary_html_str(df_for_summmary_html, include_golden_signal_dropdown=True, ignored_file_list=[], processed_file_list=[]))
