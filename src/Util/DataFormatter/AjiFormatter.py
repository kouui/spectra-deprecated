
import string
import collections

from ..External import mendeleev, roman
#from ..Nist import LevelQuery
from .. import PandasUtil
from .. import Logger
from .. import ListUtil

import pandas as pd

import os

from . import LevelFormatter

#-----------------------------------------------------------------------------

Log = Logger.MyLogger(verbose=True)

#-----------------------------------------------------------------------------

def _read_nist_line_csv(csv_path : str):

    r""" """
    df = pd.read_csv(csv_path, index_col=0, na_values="").fillna('-')
    Log.info( f"loaded Level table from {csv_path}" )

    return df

def _filter_df_line_stage(df_stage, df_line_stage):

    df_line_stage["match"] = False
    assert df_line_stage.iloc[:,-1].name == 'match'

    search_table = ListUtil.pack_multi( df_stage["configuration"].tolist(), df_stage["term"].tolist(), df_stage["J"].tolist() )
    upper_table = ListUtil.pack_multi( df_line_stage["configuration_upper"].tolist(),
                                       df_line_stage["term_upper"].tolist(),
                                       df_line_stage["J_upper"].tolist() )
    lower_table = ListUtil.pack_multi( df_line_stage["configuration_lower"].tolist(),
                                       df_line_stage["term_lower"].tolist(),
                                       df_line_stage["J_lower"].tolist() )

    n_row = len(df_line_stage)
    for i in range(n_row):
        if lower_table[i] in search_table and upper_table[i] in search_table:
            df_line_stage.iloc[i, -1] = True

    return df_line_stage[ df_line_stage["match"]==True ].drop(["match",], axis=1)

def _nist_csv_to_df( csv_path : str, line_csv_path : str, level_info : dict ):
    r""" """

    df = LevelFormatter._read_nist_csv(csv_path)
    df_line = _read_nist_line_csv(line_csv_path)

    stages = [ item for item in level_info.keys() ]

    n_stage = len( stages )

    df_line_all = None
    count = 0

    ground_conf = {}
    for i, stage in enumerate(stages):

        df_stage = PandasUtil.get_sub_df( df, 'stage', stage, _sort_key='E[eV]', _as=True )
        kk = level_info[stage]["n_level"]
        df_stage, _ = LevelFormatter._get_topk_df(df_stage, kk)
        ground_conf[stage] = df_stage["configuration"].iloc[0]

        df_line_stage = PandasUtil.get_sub_df( df_line, 'stage', stage )
        df_line_stage = _filter_df_line_stage(df_stage, df_line_stage)
        k = len(df_line_stage)


        if df_line_all is None:
            df_line_all = df_line_stage
        else:
            df_line_all = df_line_all.append( df_line_stage )
        count += k
        Log.info(f"added {k} Aji coefficients for ionization stage {stage}")

    df_line_all.reset_index( inplace=True )

    return df_line_all, ground_conf

def _nist_df_to_content(df : pd.DataFrame, element : str, ground_conf : dict):

    titles = ("conf_i", "term_i", "J_i", "conf_j", "term_j", "J_j", "Aji[s^-1]", "Wavelength[AA]", "stage")
    #s  = f"#   {titles[0]:<12s}  {titles[1]:<6s}  {titles[2]:<9s}  "
    #s += f"{titles[3]:<12s}  {titles[4]:<6s}  {titles[5]:<9s}"
    #s += f"{title[6]:1.3E}  {title[7]:4.2E}  {title[8]:5s}\n"
    s  = f"#   {titles[0]:<12s}  {titles[1]:<6s}  {titles[2]:<9s}  "
    s += f"{titles[3]:<12s}  {titles[4]:<6s}  {titles[5]:<9s}"
    s += f"{titles[6]:<9s}  {titles[7]:<14s}  {titles[8]:5s}\n"

    n_row = len(df)

    for k in range(n_row):

        row = df.iloc[k]
        _index, _conf_i, _term_i, _J_i, _conf_j, _term_j, _J_j, _Aji, _wave, _type, _stage, _z, _zz, _zzz, _v, _vv, _vvv = row.tolist()

        _conf = ground_conf[_stage]
        if k == 0:
            conf_prefix = LevelFormatter._get_conf_prefix(_stage, element, _conf)
            s += f"prefix    {conf_prefix}\n"
        else:
            conf_prefix = LevelFormatter._precess_conf_prefix(df["stage"].iloc[k-1], _stage, element, _conf)
            if conf_prefix is not None:
                s += f"prefix    {conf_prefix}\n"

        s += f"    {_conf_i:<12s}  {_term_i:<6s}  {_J_i:<9s}  "
        s += f"{_conf_j:<12s}  {_term_j:<6s}  {_J_j:<9s}"
        s += f"{_Aji:1.3E}  {_wave:1.5E}     {roman.fromRoman(_stage):5d}"


        if k != n_row-1:
            s += '\n'

    return s


def from_nist_csv( template_path, context ):
    r""" """

    context = collections.OrderedDict( context )

    with open(template_path, 'r') as f:
        template = string.Template(f.read())

    element = context["element"]

    df, ground_conf = _nist_csv_to_df(context["csv_path"],context["line_csv_path"], context["level_info"])
    content = _nist_df_to_content(df, element, ground_conf)

    context_out = {
        'table_content' : content,
    }
    output = template.substitute( context_out )

    if not os.path.exists(context["out_folder"]):
        os.makedirs( context["out_folder"] )
    out_path = os.path.join(context["out_folder"], context["out_file"])

    with open(out_path, 'w') as f:
        f.write( output )

    Log.info( f"saved as {out_path}" )
