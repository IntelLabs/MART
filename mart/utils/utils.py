from typing import Optional

__all__ = [
    "flatten_dict",
]


def flatten_dict(d, delimiter="."):
    def get_dottedpath_items(d: dict, parent: Optional[str] = None):
        """Get pairs of the dotted path and the value from a nested dictionary."""
        for name, value in d.items():
            path = f"{parent}{delimiter}{name}" if parent else name
            if isinstance(value, dict):
                yield from get_dottedpath_items(value, parent=path)
            else:
                yield path, value

    ret = {}
    for key, value in get_dottedpath_items(d):
        if key in ret:
            raise KeyError(f"Key collision when flattening a dictionary: {key}")
        ret[key] = value

    return ret
