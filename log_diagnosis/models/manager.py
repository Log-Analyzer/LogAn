from abc import ABC, abstractmethod
from typing import Type, Dict, Optional
import importlib.util
import os


class ModelTemplate(ABC):
    """
    Abstract base class that all models must implement.
    
    To create a custom model, extend this class and implement all abstract methods.
    
    Example:
        class MyCustomModel(ModelTemplate):
            def __init__(self, model_path: str = "default"):
                self.model_path = model_path
            
            def init_model(self):
                self.model = load_model(self.model_path)
            
            def classify_golden_signal(self, input: list[str], batch_size: int = 32):
                return self.model.predict_gs(input)
            
            def classify_fault_category(self, input: list[str], batch_size: int = 32):
                return self.model.predict_fault(input)
    """
    
    @abstractmethod
    def init_model(self):
        """Initialize the model. Called once after construction."""
        pass

    @abstractmethod
    def classify_golden_signal(self, input: list[str], batch_size: int = 32) -> list[dict]:
        """
        Classify input texts into golden signal categories.
        
        Args:
            input: List of log text strings to classify.
            batch_size: Batch size for inference.
        
        Returns:
            List of dictionaries with 'labels' and 'scores' keys.
        """
        pass

    @abstractmethod
    def classify_fault_category(self, input: list[str], batch_size: int = 32) -> list[dict]:
        """
        Classify input texts into fault categories.
        
        Args:
            input: List of log text strings to classify.
            batch_size: Batch size for inference.
        
        Returns:
            List of dictionaries with 'labels' and 'scores' keys.
        """
        pass


class ModelRegistry:
    """
    Central registry for custom model classes.
    
    Supports loading models from external Python scripts using the format:
        "<path_to_script>:<class_name>"
    
    Example:
        # Register a model from external script
        ModelRegistry.register_from_path("my_model", "/path/to/script.py:MyModelClass")
        
        # Or using the shorthand
        ModelRegistry.register_from_path("my_model", "script.py:MyModelClass")
        
        # Get the registered model class
        model_cls = ModelRegistry.get("my_model")
        model = model_cls()
    """
    
    _registry: Dict[str, Type[ModelTemplate]] = {}
    
    @classmethod
    def parse_model_path(cls, model_path: str) -> tuple[str, str]:
        """
        Parse the model path string into script path and class name.
        
        Args:
            model_path: String in format "<script_path>:<class_name>"
        
        Returns:
            Tuple of (script_path, class_name)
        
        Raises:
            ValueError: If the format is invalid.
        """
        if ':' not in model_path:
            raise ValueError(
                f"Invalid model path format: '{model_path}'. "
                f"Expected format: '<path_to_script>:<class_name>'"
            )
        
        # Split only on the last colon to handle Windows paths like C:\path\script.py:Class
        parts = model_path.rsplit(':', 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid model path format: '{model_path}'. "
                f"Expected format: '<path_to_script>:<class_name>'"
            )
        
        script_path, class_name = parts
        
        if not script_path.strip():
            raise ValueError("Script path cannot be empty")
        if not class_name.strip():
            raise ValueError("Class name cannot be empty")
        
        return script_path.strip(), class_name.strip()
    
    @classmethod
    def load_class_from_script(cls, script_path: str, class_name: str) -> Type[ModelTemplate]:
        """
        Dynamically load a class from a Python script file.
        
        Args:
            script_path: Path to the Python script file.
            class_name: Name of the class to load from the script.
        
        Returns:
            The loaded class (must be a subclass of ModelTemplate).
        
        Raises:
            FileNotFoundError: If the script file doesn't exist.
            AttributeError: If the class is not found in the script.
            TypeError: If the class is not a subclass of ModelTemplate.
        """
        # Resolve to absolute path
        script_path = os.path.abspath(os.path.expanduser(script_path))
        
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Script not found: {script_path}")
        
        if not script_path.endswith('.py'):
            raise ValueError(f"Script must be a Python file (.py): {script_path}")
        
        # Create a unique module name based on the script path
        module_name = f"custom_model_{os.path.basename(script_path).replace('.py', '')}_{id(script_path)}"
        
        # Load the module
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load module spec from: {script_path}")
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get the class from the module
        if not hasattr(module, class_name):
            available_classes = [
                name for name in dir(module) 
                if isinstance(getattr(module, name, None), type)
            ]
            raise AttributeError(
                f"Class '{class_name}' not found in {script_path}. "
                f"Available classes: {available_classes}"
            )
        
        model_cls = getattr(module, class_name)
        
        # Validate it's a proper subclass
        if not isinstance(model_cls, type) or not issubclass(model_cls, ModelTemplate):
            raise TypeError(
                f"Class '{class_name}' must be a subclass of ModelTemplate. "
                f"Got: {type(model_cls)}"
            )
        
        return model_cls
    
    @classmethod
    def register_from_path(cls, name: str, model_path: str) -> Type[ModelTemplate]:
        """
        Register a model class from an external script using path format.
        
        Args:
            name: Unique name to register the model under.
            model_path: String in format "<path_to_script>:<class_name>"
        
        Returns:
            The registered model class.
        
        Example:
            ModelRegistry.register_from_path("my_model", "/path/to/model.py:MyModelClass")
            ModelRegistry.register_from_path("bert_model", "./models/bert.py:BertClassifier")
        """
        script_path, class_name = cls.parse_model_path(model_path)
        model_cls = cls.load_class_from_script(script_path, class_name)
        cls._registry[name] = model_cls
        return model_cls
    
    @classmethod
    def register_class(cls, name: str, model_cls: Type[ModelTemplate]):
        """
        Programmatically register a model class directly.
        
        Args:
            name: Unique name to register the model under.
            model_cls: The model class (must be subclass of ModelTemplate).
        """
        if not isinstance(model_cls, type) or not issubclass(model_cls, ModelTemplate):
            raise TypeError(f"Model class must be a subclass of ModelTemplate, got {model_cls}")
        cls._registry[name] = model_cls
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[ModelTemplate]]:
        """Get a registered model class by name."""
        return cls._registry.get(name)
    
    @classmethod
    def list_registered(cls) -> list[str]:
        """List all registered custom model names."""
        return list(cls._registry.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a model name is registered."""
        return name in cls._registry
    
    @classmethod
    def unregister(cls, name: str) -> bool:
        """Remove a model from the registry."""
        if name in cls._registry:
            del cls._registry[name]
            return True
        return False
    
    @classmethod
    def clear(cls):
        """Clear all registered models."""
        cls._registry.clear()
