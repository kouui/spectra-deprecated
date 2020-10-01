

import string
import collections

from ..External import mendeleev, roman
from ..Nist import LevelQuery
from .. import PandasUtil
from .. import Logger

import pandas as pd

from ... import Constants as Cst

import re
import os

#-----------------------------------------------------------------------------

Log = Logger.MyLogger(verbose=True)

#-----------------------------------------------------------------------------

def _read_nist_csv(csv_path : str):
    r""" """
    dtype = {
        "configuration" : 'object',
        "term" : "object",
        "J" : "object",
        "g" : "uint32",
        "E[eV]" : "float64",
        "stage" : "object",
        "origin_configuration" : 'object',
        "origin_term" : "object",
        "origin_J" : "object",
    }
    df = pd.read_csv(csv_path, index_col=0, dtype=dtype, na_values="").fillna('-')
    Log.info( f"loaded Level table from {csv_path}" )

    return df

def _get_topk_df(df : pd.DataFrame, kk : int):

    #-- check term belongings
    df_conf = df["configuration"]
    df_term = df["term"]
    k = kk
    while df_conf.iloc[k-1]==df_conf.iloc[k] and df_term.iloc[k-1]==df_term.iloc[k]:
        k += 1

    return df.iloc[:k], k

def _nist_csv_to_df( csv_path : str, level_info : dict ):
    r""" """

    df = _read_nist_csv(csv_path)

    stages = [ item for item in level_info.keys() ]

    n_stage = len( stages )
    #Log.info( f"Your atomic model has ionization stages : {stages}" )

    df_all = None
    count = 0
    for i, stage in enumerate(stages):
        df_stage = PandasUtil.get_sub_df( df, 'stage', stage, _sort_key='E[eV]', _as=True )
        kk = level_info[stage]["n_level"]
        df_stage, k = _get_topk_df(df_stage, kk)

        if df_all is None:
            df_all = df_stage
        else:
            df_all = df_all.append( df_stage )
        count += k
        Log.info(f"added {k} Levels from ionization stage {stage}")

        if i==n_stage-1 and roman.nextRoman(stage) not in level_info.keys():
            if level_info[stage]["has_continuum"]:
                stage_cont = roman.nextRoman(stage)
                df_cont = PandasUtil.get_sub_df( df, 'stage', stage_cont, _sort_key='E[eV]', _as=True )
                df_cont = df_cont.iloc[0]
                df_all = df_all.append( df_cont )
                count += 1
                Log.info( f"added comtinuum from {stage_cont}" )

    df_all.reset_index( inplace=True )
    df_all["E[eV]"] -= df_all["E[eV]"].iloc[0]

    return df_all, count

def _conf_to_n(s):

    ss = s.split('.')[-1]
    return ss[0]

def _term_to_L_and_2Splus1(s):

    is_match = re.match(r'([0-9]+)[A-Z]', s)

    if is_match is None:
        return '-', '-'
    else:
        return str( Cst.L_s2i[ s[-1] ] ), s[:-1]

def _get_conf_prefix(_conf, element):
    pass


def _get_conf_prefix(current_stage, element, conf):
    conf_full = LevelQuery.get_full_configuration(element + ' ' + current_stage)
    conf_prefix = LevelQuery.get_configuration_prefix(conf, conf_full)
    return conf_prefix

def _precess_conf_prefix(previous_stage, current_stage, element, conf):

    if roman.fromRoman(current_stage) > roman.fromRoman(previous_stage):
        conf_prefix = _get_conf_prefix(current_stage, element, conf)
    else:
        conf_prefix = None

    return conf_prefix




def _nist_df_to_content(df : pd.DataFrame, element : str):

    titles = ("conf", "term", "J", "n", "L", "2S+1", "g", "stage", "E[eV]")
    #s  = f"#   {titles[0]:<12s}  {titles[1]:<6s}  {titles[2]:<9s}  {titles[3]:<2s}  {titles[4]:<2s}  "
    #s += f"{titles[5]:<4s}  {titles[6]:<6d}  {titles[7]:<5s}  {titles[8]:1.7E+}"
    s  = f"#   {titles[0]:<12s}  {titles[1]:<6s}  {titles[2]:<9s}  {titles[3]:<2s}  {titles[4]:<2s}  "
    s += f"{titles[5]:<4s}  {titles[6]:<6s}  {titles[7]:<5s}  {titles[8]:<12s}\n"
    n_row = len(df)

    for k in range(n_row):

        row = df.iloc[k]
        _index, _conf, _term, _J, _g, _E, _stage, _z, _zz, _zzz = row.tolist()
        _n = _conf_to_n( _conf )
        _L, _2Sp1 = _term_to_L_and_2Splus1( _term )

        if k == 0:
            conf_prefix = _get_conf_prefix(_stage, element, _conf)
            s += f"prefix    {conf_prefix}\n"
        else:
            conf_prefix = _precess_conf_prefix(df["stage"].iloc[k-1], _stage, element, _conf)
            if conf_prefix is not None:
                s += f"prefix    {conf_prefix}\n"

        s += f"    {_conf:<12s}  {_term:<6s}  {_J:<9s}  {_n:<2s}  {_L:<2s}  "
        s += f"{_2Sp1:<4s}  {_g:<6d}  {roman.fromRoman(_stage):<5d}  {_E:1.7E}"


        if k != n_row-1:
            s += '\n'

    return s


def from_nist_csv( template_path, context ):
    r""" """

    context = collections.OrderedDict( context )

    with open(template_path, 'r') as f:
        template = string.Template(f.read())

    element = context["element"]

    df, count = _nist_csv_to_df(context["csv_path"], context["level_info"])
    content = _nist_df_to_content(df, element)


    context_out = {
        "title" : context["title"],
        "atomic_number" : f"{mendeleev.element(element).atomic_number}",
        "element_symbol" : element,
        'n_level' : f"{count}",
        'table_content' : content,
    }
    output = template.substitute( context_out )

    if not os.path.exists(context["out_folder"]):
        os.makedirs( context["out_folder"] )
    out_path = os.path.join(context["out_folder"], context["out_file"])

    with open(out_path, 'w') as f:
        f.write( output )

    Log.info( f"saved as {out_path}" )
