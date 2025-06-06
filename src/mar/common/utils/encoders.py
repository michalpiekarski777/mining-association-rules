import json

import numpy as np


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, frozenset):
            return list(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
