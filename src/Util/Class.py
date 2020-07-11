


import numpy as np

def filter_array_type(_type):
    r"""
    """
    if _type in ("ndarray",):
        return "array"
    else :
        return _type

def help(_cls):
    r"""
    """

    _attrs = vars(_cls)
    _s = f"{'name':<25s}    {'type':<15s}    {'value/len/shape':15s}\n"
    _s += "-" * 70 + '\n'

    for _name, _value in _attrs.items():


        _type = type(_value)
        if _type in (np.dtype, ):
            continue

        _s += f"{_name:25s}    {filter_array_type(_type.__name__):15s}"

        if _type in (int, float, complex):
            _s += f"    v: {_value}\n"

        elif _type in (str,):
            if len(_value) > 10:
                _s += f"    v: {_value[:10]+'...'}\n"
            else:
                _s += f"    v: {_value}\n"

        elif _type in (dict, list, tuple):
            _s += f"    l: {len(_value)}\n"

        elif _type in (np.recarray,):
            _s += "    ---\n"
            _dtype = _value.dtype
            for _arr_name in _dtype.names:
                _s += f"|-> {_arr_name:<21s}    {'array':15s}"
                _s += f"    s: {_value[_arr_name].shape}\n"

        elif _type in (np.ndarray,):
            _s += f"    s: {_value.shape}\n"

        else:
            _s += "\n"


    print(_s)
