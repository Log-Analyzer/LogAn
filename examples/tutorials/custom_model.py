'''
Custom Model Example for Logan Log Diagnosis
=============================================

This module demonstrates how to create a custom classification model for use with
the Logan log diagnosis system. Custom models allow you to use your own ML models
instead of the built-in zero-shot classifiers.

Requirements
------------
1. Your custom model class must extend `ModelTemplate` from `log_diagnosis.models.manager`
2. You must implement three required methods:
   - `init_model()`: Initialize/load your model (called once after construction)
   - `classify_golden_signal()`: Classify logs into golden signal categories
   - `classify_fault_category()`: Classify logs into fault categories

Usage
-----
Run the log diagnosis with your custom model using:

    uv run logan analyze \
        -f "examples/Linux_2k.log" \
        -o "tmp/debug" \
        --model-type custom \
        --model "examples/tutorials/custom_model.py:GLiNERModel" \
        --clean-up

If you are running this on a container, you can use the following command:

    podman run --rm \
        -v ./examples/tutorials:/data/extra/:z \
        -v ./examples/:/data/input/:z \
        -v ./tmp/output/:/data/output/:z \
        -e LOGAN_INPUT_FILES="/data/input/Linux_2k.log" \
        -e LOGAN_OUTPUT_DIR=/data/output/ \
        -e LOGAN_MODEL_TYPE=custom \
        -e LOGAN_MODEL="/data/extra/custom_model.py:GLiNERModel" \
        logan

The `--model` argument format is: "<path_to_script>:<class_name>"

Examples:
    # Using relative path
    --model "./examples/tutorials/custom_model.py:GLiNERModel"
    
    # Using absolute path
    --model "/home/user/models/my_model.py:MyCustomClassifier"
'''

from gliner2 import GLiNER2
from logan.log_diagnosis.models.manager import ModelTemplate  # Base class for all custom models


class GLiNERModel(ModelTemplate):
    """
    A custom model implementation using GLiNER2 for log classification.
    
    GLiNER2 is a zero-shot named entity recognition and classification model
    that can be used for text classification tasks without task-specific training.
    
    This implementation demonstrates:
    - Single-label classification for golden signals
    - Multi-label classification for fault categories
    - Batch processing for efficient inference
    
    Attributes:
        extractor (GLiNER2): The GLiNER2 model instance for classification.
    """
    
    def init_model(self):
        """
        Initialize the GLiNER2 model.
        
        This method is called once after the model class is instantiated.
        Use this to load model weights, initialize tokenizers, or set up
        any resources needed for inference.
        
        Note:
            - This is separate from __init__ to allow the framework to control
              when expensive model loading happens
        """
        # Load the pre-trained GLiNER2 model from HuggingFace
        self.extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

    def classify_golden_signal(self, input: list[str], batch_size: int = 32) -> list[dict]:
        """
        Classify log lines into golden signal categories.
        
        Golden signals are the four key metrics from SRE practices:
        latency, traffic, errors, and saturation. This implementation
        extends it with 'information' and 'availability' for comprehensive
        log classification. You can customize the schema to include more categories.
        
        Args:
            input (list[str]): List of log text strings to classify.
                Each string is a single log line or log message.
            batch_size (int, optional): Number of samples to process in each batch.
                Larger batches are faster but use more memory. Defaults to 32.
        
        Returns:
            list[dict]: A list of dictionaries, one per input log line.
                Each dictionary contains:
                - 'labels' (list[str]): Single-element list with the predicted category
                - 'scores' (list[float]): Single-element list with confidence score (0-1)
        """
        # Define the classification schema with all golden signal categories
        # Each category has a description to help the model understand the intent
        schema = self.extractor.create_schema().classification(
            "golden_signal",
            {
                "information": "Classifies log lines that provide general operational details, status updates, or other non-critical messages.",
                "error": "Classifies log lines that indicate an error condition, malfunction, or abnormal system event.",
                "availability": "Classifies log lines related to system uptime, downtime, service availability, or accessibility.",
                "latency": "Classifies log lines that refer to delays, response times, or performance lag within the system.",
                "saturation": "Classifies log lines that signify resource exhaustion, capacity issues, or overloaded subsystems.",
                "traffic": "Classifies log lines describing system throughput, number of requests, connections, or data flow.",
            }
        )

        # Perform batch inference on all input logs
        results = self.extractor.batch_extract(
            input,
            schema,
            batch_size=batch_size,
            threshold=0.5,
            include_confidence=True,
        )

        # Transform results to the expected output format
        # The framework expects a list of dicts with 'labels' and 'scores' keys
        formatted_result = []
        for result in results:
            # Extract the golden_signal classification from results
            classification_result = result['golden_signal']
            
            # Format as expected: single label and score in lists
            # This is single-label classification (one label per log)
            formatted_result.append({
                'labels': [classification_result['label']],
                'scores': [classification_result['confidence']]
            })
        
        return formatted_result

    def classify_fault_category(self, input: list[str], batch_size: int = 32) -> list[dict]:
        """
        Classify log lines into fault categories (multi-label classification).
        
        Fault categories identify the domain or type of issue a log relates to.
        Unlike golden signals, a single log can belong to multiple fault categories
        (e.g., a log about network authentication failure could be both 'network' 
        and 'authentication').
        
        Args:
            input (list[str]): List of log text strings to classify.
                Each string is a single log line or log message.
            batch_size (int, optional): Number of samples to process in each batch.
                Larger batches are faster but use more memory. Defaults to 32.
        
        Returns:
            list[dict]: A list of dictionaries, one per input log line.
                Each dictionary contains:
                - 'labels' (list[str]): List of predicted category labels (can be multiple)
                - 'scores' (list[float]): Corresponding confidence scores for each label
        """
        # Define multi-label classification schema for fault categories
        # multi_label=True allows multiple categories per log
        schema = self.extractor.create_schema().classification(
            "fault_category",
            {
                "io": "Classifies log lines that refer to input/output operations, file system interactions, or data transfer.",
                "authentication": "Classifies log lines that relate to user authentication, authorization, or security protocols.",
                "network": "Classifies log lines that describe network connectivity, routing, or protocol interactions.",
                "application": "Classifies log lines that refer to application-level events, process interactions, or service calls.",
                "device": "Classifies log lines that relate to hardware device status, driver activity, or peripheral operations.",
            },
            multi_label=True,
            cls_threshold=0.3,
        )

        # Perform batch inference with multi-label classification
        results = self.extractor.batch_extract(
            input,
            schema,
            batch_size=batch_size,
            threshold=0.3,
            include_confidence=True,
        )

        # Transform results to the expected output format
        # For multi-label, each log may have multiple labels and scores
        formatted_result = []
        for result in results:
            # Extract fault_category results from relation_extraction output
            # GLiNER2 returns multi-label results differently than single-label
            classification_results = result['relation_extraction']['fault_category']
            
            # Build lists of labels and their corresponding scores
            label_score = {
                'labels': [],
                'scores': []
            }
            
            # classification_results is a list of (label, score) tuples
            for label, score in classification_results:
                label_score['labels'].append(label)
                label_score['scores'].append(score)
            
            formatted_result.append(label_score)
        
        return formatted_result
