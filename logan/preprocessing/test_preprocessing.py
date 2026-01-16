import unittest
import os
import json
import warnings
import pandas as pd

from .preprocessing import Preprocessing, pyrbras
warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*")
class TestExtractTS(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.preprocessor = Preprocessing(debug_mode=False)
        self.rbr = pyrbras.load_model(os.path.join(os.path.dirname(__file__), 'model', 'manifest.json'))
        with open(os.path.join(os.path.dirname(__file__), 'timezones.json'), 'r') as f:
            self.timezone_dict = json.load(f)
        self.master_timestamp_list = self.preprocessor.master_timestamp_list
        self.master_format_list = self.preprocessor.master_format_list
        self.df = pd.read_csv("./test.csv")  # Ensure test_data.csv is the correct path
    
    def extract_ts(self, log_line):
        timestamp, ts = self.preprocessor.extract_ts(
            log_line,
            self.rbr,
            self.timezone_dict,
            self.master_timestamp_list,
            self.master_format_list
        )
        return timestamp, ts
    
    def test_extract_ts_cases(self):
        """Run extract_ts on all test cases from CSV and compare results."""
        passed, failed = 0, 0
        for _, row in self.df.iterrows():
            extracted_timestamp, extracted_ts = self.extract_ts(row["test_logs"])
            expected_timestamp = str(row["extracted_ts"])
            expected_ts = float(row["identified_ts"])
            if extracted_timestamp == expected_timestamp and extracted_ts == expected_ts:
                passed += 1
            else:
                failed += 1
                print(f"Failed for log: {row['test_logs']}\n Extracted: ({extracted_timestamp}, {extracted_ts}), Expected: ({expected_timestamp}, {expected_ts})\n")
        print(f"Total Passed: {passed}, Total Failed: {failed}")
        assert failed == 0, f"Some test cases failed: {failed}"

if __name__ == "__main__":
    unittest.main()