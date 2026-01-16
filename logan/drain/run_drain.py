import logging
import sys
import os
import json
import time
from drain3.template_miner import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.file_persistence import FilePersistence

class Templatizer:
    """
    The Templatizer class is responsible for mining log templates using the DRAIN3 algorithm.
    
    It processes log data, learns the template structures, and saves the computed templates to a specified path.
    The class also logs relevant statistics and errors during the mining process.
    
    Attributes:
        config_path (str): Path to the configuration file for the DRAIN3 template miner.
        df (pd.DataFrame): DataFrame storing the processed log data after mining.

    Methods:
        compute_drain_statistics(time, output_dir):
            Logs and stores the time taken for the DRAIN3 template mining process.
        
        miner(df, output_dir, template):
            Mines log templates from the given DataFrame and saves the templates to a specified path.
    """
    
    def __init__(self, config_path: str = "/Drain3/run_drain/drain3.ini", debug_mode: str = False):
        """
        Initializes the Templatizer with logging and a configuration path for the DRAIN3 miner.
        
        Args:
            config_path (str): The path to the DRAIN3 configuration file. Defaults to '/Drain3/run_drain/drain3.ini'.
            debug_mode (str): Flag to enable debug mode for additional logging and file saving. Defaults to False.
        """
        # Setup logging to output messages to stdout with INFO level
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')
        self.logger = logging.getLogger(__name__)

        # Store the path to the DRAIN3 configuration file
        self.config_path = config_path
        
        # Placeholder for the DataFrame after template mining
        self.df = None

        # Debug mode flag
        self.debug_mode = debug_mode
    
    def compute_drain_statistics(self, time_taken: float, output_dir: str):
        """
        Compute and save statistics about the DRAIN3 template mining process.
        
        Args:
            time_taken (float): The time taken for the template mining process, in seconds.
            output_dir (str): The directory where the statistics will be saved.
        """
        # Dictionary to store the time taken for the DRAIN3 mining process
        metrics = {
            'drain_templatisation_time_ms': time_taken
        }

        # Save the metrics as a JSON file in the specified output directory
        with open(os.path.join(output_dir, "metrics", "drain.json"), 'w') as writer:
            writer.write(json.dumps(metrics, indent=4))
    
    def miner(self, df, output_dir: str, template: str):
        """
        Apply the DRAIN3 template mining algorithm to the given DataFrame.
        
        This method processes log messages from the DataFrame's 'text' column, learns template patterns,
        and assigns a template cluster ID to each log message. The learned templates are saved to the specified path.

        Args:
            df (pd.DataFrame): DataFrame containing log data with a 'truncated_log' column for mining.
            output_dir (str): The directory where output files, such as statistics, will be saved.
            template (str): The file path where the mined templates will be stored.
        """
        # Record the start time of the DRAIN3 mining process
        start_time = time.time()
        self.logger.info("Starting DRAIN")

        # Load the DRAIN3 configuration from the specified path
        config = TemplateMinerConfig()
        config.load(self.config_path)

        # Set up file persistence to store the learned templates in the specified template file
        mem_persistence = FilePersistence(template)
        
        # Initialize the TemplateMiner with the loaded configuration and file persistence
        template_miner_temporary = TemplateMiner(mem_persistence, config)

        # Initialize a dictionary to store the loglines grouped by template IDs
        template_log_dict = {}

        # Attempt to mine templates by applying DRAIN to each log message in the DataFrame
        try:
            # Apply the template miner to the 'truncated_log' column and assign the resulting cluster ID
            df["test_ids"] = df.apply(
                lambda row: template_miner_temporary.add_log_message(row["truncated_log"])['cluster_id'],
                axis=1
            )
            if (self.debug_mode == "true"):
                template_log_dict = df.groupby("test_ids")["truncated_log"].agg(list).to_dict()
                with open(os.path.join(output_dir, "developer_debug_files", "matcher_output_json.json"), 'w') as json_file:
                    json.dump(template_log_dict, json_file, indent=4)

        except Exception as e:
            # Log any errors encountered during the mining process
            self.logger.error(f"Error learning templates for -1 test_ids in DataFrame: {e}")

        # Store the DataFrame with the assigned cluster IDs
        self.df = df
        
         # Compute and save statistics for the DRAIN3 mining process
        self.compute_drain_statistics((time.time() - start_time) * 1000, output_dir)