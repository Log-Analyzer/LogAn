import json
from argparse import ArgumentParser
from datetime import timedelta
from itertools import combinations
from statsmodels.tsa.stattools import grangercausalitytests
import csv
from jinja2 import Template

import pandas as pd
import numpy as np



arg_parser = ArgumentParser()
arg_parser.add_argument('--input_file', help='Input File for Causality')
arg_parser.add_argument('--product_name', help='Product name for which tool is running')
arg_parser.add_argument('--signal_map', help='Template ID to Golden Signal and Fault Category map')
arg_parser.add_argument('--template_map', help='Template ID to Template string')
arg_parser.add_argument('--output_file', help='Output HTML File for Casuality')
args = arg_parser.parse_args()

def run_granger_causality(timeseries, top_k):
    results = {}

    template_pairs = combinations(timeseries.keys(), 2)

    for template_pair in template_pairs:
        template1, template2 = template_pair

        # Make the time series stationary by differencing
        differenced_series1 = np.diff(timeseries[template1])
        differenced_series2 = np.diff(timeseries[template2])

        # Check for constant columns in differenced_series1 and differenced_series2
        if np.unique(differenced_series1).size == 1 or np.unique(differenced_series2).size == 1:
            continue
 
        data = pd.DataFrame({'Template1': differenced_series1, 'Template2': differenced_series2})
        
        try:
            result = grangercausalitytests(data, maxlag=5, verbose=False)
        except InfeasibleTestError:
            # Handle cases where the test statistic cannot be computed
            continue
        except ValueError as e:
            # Handle other potential errors
            print(f"Error occurred: {e}")
            continue
        # Check if Granger causality exists (p-value is below threshold, e.g., 0.05)
        granger_exists = any(result[lag][0]['ssr_chi2test'][1] < 0.05 for lag in result.keys())

        if granger_exists:
            results[template_pair] = result

    # Sort the results based on the p-values in ascending order
    sorted_results = sorted(results.items(), key=lambda x: np.min([x[1][lag][0]['ssr_chi2test'][1] for lag in x[1].keys()]))

    # Select the top K pairs
    top_k_results = dict(sorted_results[:top_k])

    return top_k_results

from datetime import timedelta

def create_timeseries(df):
    df = df.sort_values('timestamp')
    total_duration = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
    num_points = 1000
    
    # Calculate duration for the stride
    stride_duration = total_duration / num_points
    stride = int(stride_duration.total_seconds())
    
    # Calculate window size based on the stride duration
    window_size = stride_duration * 5

    timeseries = {}
    start_time = df['timestamp'].iloc[0]
    end_time = start_time + window_size
    while end_time <= df['timestamp'].iloc[-1]:
        window = df[(df['timestamp'] >= start_time) & (df['timestamp'] < end_time)]

        template_counts = window['template'].value_counts().to_dict()

        for template in df['template'].unique():
            count = template_counts.get(template, 0)
            if template not in timeseries:
                timeseries[template] = []
            timeseries[template].append(count)

        start_time += timedelta(seconds=stride)
        end_time = start_time + window_size

    for template in list(timeseries.keys()):
        if sum(timeseries[template]) == 0 or len(set(timeseries[template])) <= 3:
            del timeseries[template]

    return timeseries



def filter_using_gs(inferencing_file, template_to_signal_dict):
    df = pd.read_csv(inferencing_file, usecols=['test_ids', 'epoch'], quoting=csv.QUOTE_NONE, quotechar='',escapechar='\\')
    df.rename(columns={'test_ids': 'template'}, inplace=True)
    df['gs'] = df['template'].apply(lambda tid: template_to_signal_dict[tid])
    df['timestamp'] = pd.to_datetime(df['epoch'], unit='s')
    df.drop('epoch', axis=1, inplace=True)
    
    return df[df.gs != 'information']

#Window size is in seconds
def get_causal_pairs(inferencing_file, template_to_signal_dict, window_size=3600, stride = 900, top_k=10):
    causal_pairs = []

    df_timestamp_template = filter_using_gs(inferencing_file, template_to_signal_dict)
    print(df_timestamp_template)

    print('Started timeseries creation')
    timeseries = create_timeseries(df_timestamp_template) #, window_size, stride)
    causality_results = run_granger_causality(timeseries, top_k)

    print('Causality over')
    for key, value in causality_results.items():
        b_tmp = key[0]
        a_tmp = key[1]
        causal_pairs.append((a_tmp, b_tmp))

    return causal_pairs


# def run_temporal_evolution(inferencing_file, template_to_signal_file, window_size=43200):
#     with open(template_to_signal_file, 'r') as reader:
#         template_to_signal_dict = {
#             int(tid) : gs
#             for tid, (gs, _) in json.load(reader).items()
#         }

#     df = filter_using_gs(inferencing_file, template_to_signal_dict)
#     df = df.set_index('timestamp').drop('template', axis=1)

#     df_count = pd.get_dummies(df['gs']).resample(f'{window_size}S').sum()
#     df_count = df_count.reset_index()
#     df_count['start_time'] = df_count['timestamp'].astype(str)
#     df_count.drop('timestamp', axis=1, inplace=True)

#     return {
#         'data': df_count.to_dict('records'),
#         'message': 'Per Hour GS Distribution',
#         'status': 'success'
#     }



def run_temporal_evolution(inferencing_file, template_to_signal_file, num_rows=30):
    with open(template_to_signal_file, 'r') as reader:
        template_to_signal_dict = {
            int(tid): gs
            for tid, (gs, _) in json.load(reader).items()
        }

    df = filter_using_gs(inferencing_file, template_to_signal_dict)
    df = df.set_index('timestamp').drop('template', axis=1)

    # Calculate total duration covered by timestamps
    total_duration = df.index.max() - df.index.min()
    
    # Calculate window size based on desired number of rows
    window_size = total_duration / num_rows

    df_count = pd.get_dummies(df['gs']).resample(f'{window_size.seconds}S').sum()
    df_count = df_count.reset_index()
    df_count['start_time'] = df_count['timestamp'].astype(str)
    df_count.drop('timestamp', axis=1, inplace=True)

    return {
        'data': df_count.to_dict('records'),
        'message': 'Per Hour GS Distribution',
        'status': 'success'
    }

def run_causality(inferencing_file, template_to_signal_file, template_map):
    # start_time_post = time.time()

    with open(template_to_signal_file, 'r') as reader:
        template_to_signal_dict = {
            int(tid) : gs
            for tid, (gs, _) in json.load(reader).items()
        }
    
    with open(template_map, 'r') as reader:
        template_to_rep_log = {int(id): log for id, log in json.load(reader).items()}

    #Run Causality
    causal_pairs = get_causal_pairs(inferencing_file, template_to_signal_dict)
    
    #Create nodes and edges
    nodes_arr = []
    edges_arr = []
    node_ids_already = []

    for pair in causal_pairs:
        src = int(pair[0])
        trgt = int(pair[1])

        if src not in node_ids_already:
            gs = template_to_signal_dict[src]
            node_src = {'id': src, 'label': template_to_rep_log[src], 'gs': gs}
            node_ids_already.append(src)
            nodes_arr.append(node_src)
    
        if trgt not in node_ids_already:
            gs = template_to_signal_dict[trgt]
            node_trgt = {'id': trgt, 'label': template_to_rep_log[trgt], 'gs': gs}
    
            node_ids_already.append(trgt)
            nodes_arr.append(node_trgt)
    
        edge = {'source':src, 'target':trgt}
        edges_arr.append(edge)
    
    #print(nodes_arr)
    graph = {    
        'Nodes': nodes_arr,
        'Edges': edges_arr,
    }

    return graph

def render_template(graph, bar_chart):
    with open('templates/causality.html', 'r') as reader:
        html = reader.read()

    html_template = Template(html)
    rendered_template = html_template.render(graph_nodes=graph['Nodes'], graph_edges=graph['Edges'], temporal_evolution=bar_chart, title=f'{args.product_name}_Causality')
    
    with open(f'{args.output_file}', 'w') as writer:
        writer.write(rendered_template)
    

if __name__ == "__main__":
    # print(run_causality(args.input_file, args.signal_map))
    graph = run_causality(args.input_file, args.signal_map, args.template_map)
    bar_chart = run_temporal_evolution(args.input_file, args.signal_map)
    
    render_template(graph, bar_chart)