

def get_sub_df( _df, _key, _value, _sort_key=None, _as=True):

    #-- filter
    _df1 = _df[_df[_key]==_value]

    #-- sort
    if _sort_key is not None:
        _df1 = _df1.sort_values(by=_sort_key, ascending=_as)

    return _df1
