


import numpy as np

import inspect

def type_string(_type, _value):
    r"""
    """
    if _type in ("ndarray",):

        return "array"

    elif _type in ("list",):

        _is_single_type = True
        _type0 = type(_value[0])
        for _var in _value[1:]:
            if not isinstance(_var, _type0):
                _is_single_type = False
                break
        if _is_single_type:
            if _type0.__name__ in ("ndarray",):
                return "list of array"
            else:
                return f"list of {_type0.__name__}"
        else:
            return _type

    else :

        return _type

def attribute_string_recarray(_value):
    r"""
    """
    _s = "    ---\n"
    _dtype = _value.dtype
    for _arr_name in _dtype.names:
        _s += f"  |-> {_arr_name:<19s}    {_value[_arr_name].dtype.name+' array':15s}"
        _s += f"    s: {_value[_arr_name].shape}\n"

    return _s

def print_attribute(_attrs):
    r"""
    """
    _s = ""
    _class_name_list = []
    for _name, _value in _attrs.items():


        _type = type(_value)
        if _type in (np.dtype, ):
            continue

        if _type.__name__ in ("Collisional_Transition", "Photoionization", "WavelengthMesh","RadiativeLine"):
            _class_name_list.append( _name )
            continue

        if _name[0] == "_":
            continue

        _s += f"{_name:25s}    {type_string(_type.__name__, _value):15s}"

        if _type in (int, float, complex, bool):
            _s += f"    v: {_value}\n"

        elif _type in (str,):
            if len(_value) > 10:
                _s += f"    v: {_value[:10]+'...'}\n"
            else:
                _s += f"    v: {_value}\n"

        elif _type in (dict, tuple, list):
            _s += f"    l: {len(_value)}\n"

        elif _type in (np.recarray,):
            _s += attribute_string_recarray(_value)

        elif _type in (np.ndarray,):
            _s += f"    s: {_value.shape}\n"

        else:
            _s += "\n"


    print(_s)

    return _class_name_list

def help_recarray(_recarray):
    r"""
    """
    _s = f"{'name':<25s}    {'type':<15s}    {'value/len/shape':15s}\n"
    _s += "-" * 70 + '\n'
    _s += f"{' ':25s}    {'recarray':15s}"
    _s += attribute_string_recarray(_recarray)

    print(_s)


def help_attribute(_obj):
    r"""
    """

    _attrs = vars(_obj)

    _s  = "Attributes\n"
    _s += "-" * 70 + '\n'
    _s += f"{'name':<25s}    {'type':<15s}    {'value/len/shape':15s}\n"
    _s += "-" * 70 + '\n'

    print(_s)

    _class_name_list = print_attribute(_attrs)

    if len(_class_name_list)>0:
        for _name in _class_name_list:
            _sub_obj = _attrs[_name]
            _s = f"{_obj.__class__.__name__}.{_name}\n"
            _s += "-" * 30 + '\n'
            print(_s)
            _sub_attrs = vars(_sub_obj)
            _class_name_list0 = print_attribute(_sub_attrs)
            assert len(_class_name_list0)==0



def get_method_doc_description(_doc):
    r"""
    """
    _idx = _doc.find("Parameters")
    if _idx == -1:
        return _doc.rstrip()
    else:
        return _doc[:_idx].rstrip()

def print_method(_obj):
    r"""
    """
    _s = ""
    for _m in inspect.getmembers(_obj, inspect.ismethod):
        _name = _m[0]
        _method = _m[1]

        if _name[0] == "_":
            continue

        #_obj.__class__.__name__ to get the _obj name as a string
        _s += f"{_name}\n"

        if _method.__doc__ is not None:
            _s += get_method_doc_description(_method.__doc__) + "\n\n"
        else :
            _s += "\n"

    print(_s)

def help_method(_obj):
    r"""
    """
    _s  = "Methods\n"
    _s += "-" * 70 + '\n'

    print(_s)

    print_method(_obj)

def help(_obj):
    r"""
    """
    help_attribute(_obj)
    help_method(_obj)
