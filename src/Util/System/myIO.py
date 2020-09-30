import pickle, json


def _dump_json(_data, _fname, _indent=2, _ensure_ascii=False):

    with open(_fname, 'w') as _f:
        json.dump(_data, _f, indent=_indent, ensure_ascii=_ensure_ascii)

def _load_json(_fname):

    with open(_fname) as _f:
        _data = json.load(_f)

    return _data

def _dump_pickle(_data, _fname):

    with open(_fname, 'wb') as _f:
        pickle.dump(_data, _f, protocol=pickle.HIGHEST_PROTOCOL)

def _load_pickle(_fname):

    with open( _fname , 'rb') as _f:
        _data= pickle.load(_f)

    return _data
