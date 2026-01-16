import os
import json
from glob import glob
from datetime import datetime
from argparse import ArgumentParser

from es import get_stats_index, create_document
from utils import get_product_name_from_key, get_case_number

def generate_stats_doc(dir, product_key, xml_file, stage):
    all_metrics_fp = glob(os.path.join(dir, "**.json"))
    doc = {}
    
    for fp in all_metrics_fp:
        with open(fp, 'r') as reader:
            tmp = json.load(reader)
        
        doc.update(tmp)
    
    total_processing_time_ms = 0
    for key, value in doc.items():
        if key.endswith("time_ms"):
            total_processing_time_ms += float(value)
    
    doc['total_processing_time_ms'] = total_processing_time_ms

    doc['case_number'] = get_case_number(dir)
    doc['failed_stage'] = None if doc['success'] else stage
    doc['product_name'] = get_product_name_from_key(product_key, xml_file)
    doc['timestamp_triggered'] = datetime.now().isoformat()
    
    return doc


def send_stats(dir, product_key, xml_file, stage):
    # Example document

    doc = generate_stats_doc(dir, product_key, xml_file, stage)
    print(json.dumps(doc, indent=2))

    # Index name
    index_name = get_stats_index()
    # Create document in Elasticsearch
    try:
        result = create_document(index=index_name, document=doc)
        print("Document created with ID:")
        print(result)
    except Exception as err:
        print("Failed to send statistics to Elasticsearch")
        print(err)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Directory containing stats metadata")
    parser.add_argument("--product_key", type=str, required=True)
    parser.add_argument("--xml_file", type=str, required=True)
    parser.add_argument("--stage", type=str, default=None)
    args = parser.parse_args()

    send_stats(args.dir, args.product_key, args.xml_file, args.stage)