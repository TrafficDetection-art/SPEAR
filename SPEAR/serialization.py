"""
Custom JSON serialization utilities.
"""

import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types and other non-serializable objects."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, "item"):  # numpy scalar
            return obj.item()
        elif hasattr(obj, "__dict__"):
            return f"<{obj.__class__.__name__} object - not serializable>"
        else:
            try:
                return str(obj)
            except Exception:
                return f"<{type(obj).__name__} - cannot serialize>"
