from enum import Enum
from typing import Union, Optional

from logan.log_diagnosis.models.manager import ModelTemplate, ModelRegistry
from logan.log_diagnosis.models.model_zero_shot_classifer import ZeroShotModels, ModelZeroShotClassifer

AllModels = Union[ZeroShotModels, str]


class ModelType(Enum):
    ZERO_SHOT = 'zero_shot'
    SIMILARITY = 'similarity'
    CUSTOM = 'custom'


class ModelManager:
    """
    Manager for model instantiation and lifecycle.
    
    Supports built-in models and custom models loaded from external scripts.
    
    Usage:
        1. Built-in zero-shot model:
            manager = ModelManager(ModelType.ZERO_SHOT, ZeroShotModels.BART)
        
        2. Custom model from script path:
            manager = ModelManager(
                ModelType.CUSTOM, 
                model="/path/to/script.py:MyModelClass"
            )
        
        3. Custom model with constructor kwargs:
            manager = ModelManager(
                ModelType.CUSTOM,
                model="./models/custom.py:BertClassifier",
                custom_model_kwargs={"threshold": 0.8}
            )
        
        4. Pass pre-instantiated custom model:
            my_model = MyModelClass()
            manager = ModelManager(ModelType.CUSTOM, custom_model_instance=my_model)
    """
    
    def __init__(
        self,
        type: ModelType = ModelType.ZERO_SHOT,
        model: AllModels = ZeroShotModels.CROSSENCODER,
        # Custom model options
        custom_model_instance: Optional[ModelTemplate] = None,
        custom_model_kwargs: Optional[dict] = None,
    ):
        """
        Initialize the ModelManager.
        
        Args:
            type: The type of model to use (ZERO_SHOT, SIMILARITY, or CUSTOM).
            model: For built-in types, the specific model enum/string.
                   For CUSTOM type, the path in format "<script_path>:<class_name>".
            custom_model_instance: Pre-instantiated model instance (for CUSTOM type).
            custom_model_kwargs: Keyword arguments to pass to custom model constructor.
        """
        self.model: ModelTemplate
        
        if type == ModelType.ZERO_SHOT:
            self.model = ModelZeroShotClassifer(model)
            
        elif type == ModelType.SIMILARITY:
            # self.model = ModelSimilarity(model)
            raise NotImplementedError("Similarity model not yet implemented")
            
        elif type == ModelType.CUSTOM:
            self.model = self._create_custom_model(
                model_path=model if isinstance(model, str) else None,
                custom_model_instance=custom_model_instance,
                custom_model_kwargs=custom_model_kwargs or {},
            )
        else:
            raise ValueError(f"Invalid model type: {type}")

        # Initialize model
        self.model.init_model()
        print("Initialized model")
    
    def _create_custom_model(
        self,
        model_path: Optional[str],
        custom_model_instance: Optional[ModelTemplate],
        custom_model_kwargs: dict,
    ) -> ModelTemplate:
        """Create a custom model from various input methods."""
        
        # Method 1: Direct model instance provided
        if custom_model_instance is not None:
            if not isinstance(custom_model_instance, ModelTemplate):
                raise TypeError(
                    f"custom_model_instance must be an instance of ModelTemplate, "
                    f"got {type(custom_model_instance)}"
                )
            return custom_model_instance
        
        # Method 2: Load from path string (e.g., "/path/to/script.py:ClassName")
        if model_path is not None:
            # Auto-register with a generated name if not already registered
            auto_name = f"_auto_{hash(model_path)}"
            if not ModelRegistry.is_registered(auto_name):
                ModelRegistry.register_from_path(auto_name, model_path)
            model_cls = ModelRegistry.get(auto_name)
            return model_cls(**custom_model_kwargs)
        
        raise ValueError(
            "For CUSTOM model type, provide either: "
            "model='<script_path>:<class_name>' or custom_model_instance=<instance>"
        )
    
    def classify_golden_signal(self, input: list[str], batch_size: int = 32):
        """Classify input texts into golden signal categories."""
        return self.model.classify_golden_signal(input, batch_size)
    
    def classify_fault_category(self, input: list[str], batch_size: int = 32):
        """Classify input texts into fault categories."""
        return self.model.classify_fault_category(input, batch_size)


# Public API exports
__all__ = [
    'ModelManager',
    'ModelType',
    'ModelTemplate',
    'ModelRegistry',
    'AllModels',
    'ZeroShotModels',
]
