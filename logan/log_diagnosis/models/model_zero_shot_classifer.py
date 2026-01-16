from enum import Enum
from typing import Union
from logan.log_diagnosis.models.manager import ModelTemplate
from transformers import pipeline

class ZeroShotModels(Enum):
    BART = 'facebook/bart-large-mnli'
    CROSSENCODER = 'cross-encoder/nli-MiniLM2-L6-H768'

class ModelZeroShotClassifer(ModelTemplate):
    def __init__(self, model: Union[ZeroShotModels, str]):
        """
        Initializes the ModelZeroShotClassifer class.

        Args:
            model: The model to use for zero-shot classification.
                - If ZeroShotModels, the model is converted to the string value.
                - If str, the model is used as is.
        """
        super().__init__()
        if isinstance(model, ZeroShotModels):
            self.model = model.value
        else:
            self.model = model

    def init_model(self):
        self.pipe = pipeline(task='zero-shot-classification', model=self.model)

    def classify_golden_signal(self, input: list[str], batch_size: int=32):
        candidate_labels = ["information", "error", "availability", "latency", "saturation", "traffic"]
        results = self.pipe(input, candidate_labels, batch_size=batch_size)
        if isinstance(results, dict):
            # zero shot classification returns a dictionary for a single input, so we need to convert it to a list
            results = [results]
        return results
    
    def classify_fault_category(self, input: list[str], batch_size: int=32):
        candidate_labels = ["io", "authentication", "network", "application", "device"]
        results = self.pipe(input, candidate_labels, batch_size=batch_size)
        if isinstance(results, dict):
            # zero shot classification returns a dictionary for a single input, so we need to convert it to a list
            results = [results]
        return results
