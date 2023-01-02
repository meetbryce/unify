from dataclasses import replace
from multiprocessing.sharedctypes import Value
import typing
from typing import Any

XformFunc = typing.Callable[[Any, Any], Any]
# Xform funcs always get the key and value, and return either a new key or a new value

def dict_deep_transform(node: Any, value_xform=None, key_xform=None):
    """ Travereses a nested dict (one which may have values that are themselves
        dicts or lists, and returns a new dict transformed by the provided
        functions. "key_xform" can transform dict keys, and "value_xform" can
        transform values.
    """
    if isinstance(node, dict):
        res = {}
        for key, val in node.items():
            newkey = key_xform(key, val) if key_xform is not None else key
            if isinstance(val, (dict, list)):
                newval = dict_deep_transform(val, value_xform, key_xform)
            else:
                newval = value_xform(newkey, val) if value_xform is not None else val
            res[newkey] = newval
        return res
    elif isinstance(node, list):
        return [dict_deep_transform(x, value_xform, key_xform) for x in node]
    else:
        if value_xform is not None:
            return value_xform("__root__", node)
        else:
            return node


def interp_dollar_values(node: dict, replacements: dict):
    """ Searches for keys from replacements in the values of opts and replaces
        with the value from replacements. 
    """
    def str_replace_values(key, value: str):
        if not isinstance(value, str):
            return value
        else:
            newv:str = value
            for k, v in replacements.items():
                newv = newv.replace("${" + k + "}", v)
            return newv

    return dict_deep_transform(node, value_xform=str_replace_values)

def flatten_dict(target: dict):
    """ Returns a new dict with all keys,vals flattened to top level. """
    res = {}
    def grab_value(key, value):
        if not isinstance(value, dict):
            res[key] = value
        return value

    dict_deep_transform(target, value_xform=grab_value)
    return res