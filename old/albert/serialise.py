"""Serialisation of algebraic expressions.
"""

import importlib
import json


def save(expr, filename):
    """Save the expression to a file."""

    # Save the expression as JSON to the file
    with open(filename, "w") as file:
        json.dump(expr.as_json(), file, indent=4)


def load(filename):
    """Load an expression from a file."""

    # Load the JSON data from the file
    with open(filename, "r") as file:
        data = json.load(file)

    # Make a recursive function to load the data
    module_cache = {}

    def _load(data):
        # If the data is a list or tuple, load each element
        if isinstance(data, (list, tuple)):
            return [_load(value) for value in data]

        # If the data is not a dictionary, return it as is
        if not isinstance(data, dict):
            return data

        # Get the path and type of the class
        _path = data.pop("_path")
        _type = data.pop("_type")

        # Get the class from the module
        if (_path, _type) not in module_cache:
            module_cache[_path, _type] = getattr(importlib.import_module(_path), _type)
        cls = module_cache[_path, _type]

        # Load the data into the class
        data = {key: _load(value) for key, value in data.items()}
        obj = cls.from_json(data)

        return obj

    return _load(data)
