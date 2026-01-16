"""
Mock implementation of pyrbras module for log analysis pipeline.

This module provides a stub implementation that mimics the behavior of the
original pyrbras library for datetime extraction from log messages.

Authored by Cursor AI
"""

import json
import re
from datetime import datetime
from typing import Dict, Any, Optional, List

# TODO: Replace with alternate implementation/module
class MockModel:
    """Mock model class that simulates pyrbras datetime extraction behavior."""
    
    def __init__(self, manifest_path: str = None):
        """
        Initialize the mock model with manifest configuration.
        
        Args:
            manifest_path: Path to the manifest.json file
        """
        self.manifest_path = manifest_path
        self.manifest = self._load_manifest()
        
        # Common datetime patterns for extraction
        self.datetime_patterns = [
            # ISO format: 2023-12-01T10:30:45Z, 2023-12-01T10:30:45.123Z
            r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{3})?Z?',
            # Standard format: 2023-12-01 10:30:45, 2023/12/01 10:30:45
            r'\d{4}[-/]\d{2}[-/]\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d{3})?',
            # Syslog format: Dec 01 10:30:45, Dec  1 10:30:45
            r'[A-Za-z]{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}',
            # Apache format: 01/Dec/2023:10:30:45 +0000
            r'\d{2}/[A-Za-z]{3}/\d{4}:\d{2}:\d{2}:\d{2}\s*[+-]\d{4}',
            # Timestamp with brackets: [2023-12-01 10:30:45]
            r'\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\]',
            # Date only: 2023-12-01, 2023/12/01
            r'\d{4}[-/]\d{2}[-/]\d{2}',
            # Time only: 10:30:45, 10:30:45.123
            r'\d{2}:\d{2}:\d{2}(?:\.\d{3})?',
            # Unix timestamp-like: 1638360645 (must be at start or after space)
            r'(?:^|\s)\d{10}(?=\s|$)',
            # Millisecond timestamp-like: 1638360645123 (must be at start or after space)
            r'(?:^|\s)\d{13}(?=\s|$)',
            # RFC 2822 format: Fri, 01 Dec 2023 10:30:45 GMT
            r'[A-Za-z]{3},\s+\d{1,2}\s+[A-Za-z]{3}\s+\d{4}\s+\d{2}:\d{2}:\d{2}\s+[A-Z]{3}',
            # Custom format with year-day: 24165 16:48:54.18
            r'\d{2,4}\d{3}\s+\d{2}:\d{2}:\d{2}\.\d{2}',
            # Hexadecimal timestamp: 5f2c6e00
            r'\b[0-9A-Fa-f]{8}\b',
        ]
        
        # Compile patterns for better performance
        self.compiled_patterns = [re.compile(pattern) for pattern in self.datetime_patterns]
    
    def _load_manifest(self) -> Dict[str, Any]:
        """Load manifest configuration from JSON file."""
        try:
            with open(self.manifest_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Return default manifest if file doesn't exist or is invalid
            return {
                "annotator": {"version": "1.0", "key": "Demo"},
                "annotatorRuntime": "SystemT",
                "version": "1.0",
                "outputTypes": ["DateTimeOutput"]
            }
    
    def process(self, text: str, language: str = "en") -> str:
        """
        Process text to extract datetime information.
        
        Args:
            text: Input text to process
            language: Language code (default: "en")
            
        Returns:
            JSON string containing datetime annotations
        """
        datetime_matches = self._extract_datetime(text)
        
        # Create response structure matching expected format
        response = {
            "annotations": {
                "DateTimeOutput": []
            }
        }
        
        # Add found datetime matches
        for match_text, start_pos, end_pos in datetime_matches:
            annotation = {
                "span": {
                    "text": match_text,
                    "begin": start_pos,
                    "end": end_pos
                }
            }
            response["annotations"]["DateTimeOutput"].append(annotation)
        
        return json.dumps(response)
    
    def _extract_datetime(self, text: str) -> List[tuple]:
        """
        Extract datetime patterns from text.
        
        Args:
            text: Input text to search
            
        Returns:
            List of tuples containing (match_text, start_pos, end_pos)
        """
        matches = []
        
        for pattern in self.compiled_patterns:
            for match in pattern.finditer(text):
                match_text = match.group().strip()
                start_pos = match.start()
                end_pos = match.end()
                
                # Adjust positions for stripped whitespace
                if match.group().startswith(' '):
                    start_pos += 1
                if match.group().endswith(' '):
                    end_pos -= 1
                
                # Basic validation to avoid false positives
                if self._is_valid_datetime_candidate(match_text):
                    matches.append((match_text, start_pos, end_pos))
        
        # Remove duplicates and overlapping matches, keep the longest match
        matches = self._deduplicate_matches(matches)
        
        return matches
    
    def _is_valid_datetime_candidate(self, text: str) -> bool:
        """
        Perform basic validation on datetime candidates.
        
        Args:
            text: Candidate datetime string
            
        Returns:
            True if the text looks like a valid datetime
        """
        # Skip very short strings
        if len(text) < 4:
            return False
        
        # Skip strings that are all digits but too short/long for timestamps
        if text.isdigit():
            # Allow Unix timestamps (10 digits) and millisecond timestamps (13 digits)
            if len(text) in [10, 13]:
                return True
            if len(text) < 8 or len(text) > 13:
                return False
        
        # Skip hexadecimal strings that might be IDs rather than timestamps
        if re.match(r'^[0-9A-Fa-f]+$', text) and len(text) != 8:
            return False
        
        return True
    
    def _deduplicate_matches(self, matches: List[tuple]) -> List[tuple]:
        """
        Remove overlapping matches, keeping the longest one.
        
        Args:
            matches: List of (match_text, start_pos, end_pos) tuples
            
        Returns:
            Deduplicated list of matches
        """
        if not matches:
            return matches
        
        # Sort by start position
        matches.sort(key=lambda x: x[1])
        
        deduplicated = []
        for match_text, start_pos, end_pos in matches:
            # Check if this match overlaps with any existing match
            overlaps = False
            for existing_match in deduplicated:
                existing_start = existing_match[1]
                existing_end = existing_match[2]
                
                # Check for overlap
                if not (end_pos <= existing_start or start_pos >= existing_end):
                    # There's an overlap, keep the longer match
                    if len(match_text) > len(existing_match[0]):
                        # Remove the existing match and add this one
                        deduplicated.remove(existing_match)
                        break
                    else:
                        # Keep the existing match, skip this one
                        overlaps = True
                        break
            
            if not overlaps:
                deduplicated.append((match_text, start_pos, end_pos))
        
        return deduplicated


def load_model(manifest_path: str) -> MockModel:
    """
    Load a pyrbras model from a manifest file.
    
    Args:
        manifest_path: Path to the manifest.json file
        
    Returns:
        MockModel instance
    """
    return MockModel(manifest_path)


# Additional utility functions that might be expected
def get_version() -> str:
    """Get the version of the pyrbras module."""
    return "1.0.3-mock"


def get_supported_languages() -> List[str]:
    """Get list of supported languages."""
    return ["en", "es", "fr", "de", "it", "pt", "ja", "zh", "ko"]
