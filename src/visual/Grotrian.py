import matplotlib.pyplot as plt
from collections import Counter
from .. import Constants as Cst
import math

def _prepare_dict(_atom, _conf_prefix):
    r"""
    separate singlet and multiplet

        - singlet   : { (conf_clean,term) : { J : (E[eV], L, conf_origin) } }
        - multiplet : { (conf_clean,term) : { J : (E[eV], L, conf_origin) } }

    Parameters
    ----------
    _atom : AtomCls.Atom
        object of the atomic model

    _conf_prefix : str
        common configuration string of the inner shell

    Returns
    -------

    _singlet : { (conf_clean,term) : { J : (E[eV], L, conf_origin) } }

    _multiplet : { (conf_clean,term) : { J : (E[eV], L, conf_origin) } }

    _Lset : { "singlet" : {int,}, "multiplet" : {int,} }
        set of Quantum number L in integer for singlet and multiplet, respectively.
    """
    _L_prefix = len(_conf_prefix)

    #-------------------------------------------------------------------------
    # create and count list of (conf, term)
    #-------------------------------------------------------------------------
    _conf_term = list( zip( _atom._Level_info["configuration"], _atom._Level_info["term"] ) )
    _count = Counter( _conf_term )
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    # separate singlet and multiplet
    #-------------------------------------------------------------------------
    _L_prefix = len(_conf_prefix)

    _singlet = {}
    _multiplet = {}
    _Lset = {"singlet" : set(), "multiplet" : set()}
    for k in range(_atom.nLevel):

        #---------------------------------------------------------------------
        # - remove inner shell electron configuration
        # - remove all '.' in conf
        #---------------------------------------------------------------------
        _conf = _atom._Level_info["configuration"][k]
        _conf_clean = _conf[_L_prefix:].replace('.','')
        #---------------------------------------------------------------------
        _term = _atom._Level_info["term"][k]
        _J = _atom._Level_info["J"][k]
        _L = Cst.L_s2i[ _term[-1] ]

        if _count[ (_conf, _term) ] == 1:
            _d = _singlet
            _Lset["singlet"].add(_L)
        else:
            _d = _multiplet
            _Lset["multiplet"].add(_L)

        _key = (_conf_clean ,_term)
        if _key not in _d.keys():
            _d[_key] = {}
        _d[_key][_J] = ( _atom.Level.erg[k] / Cst.eV2erg_, _L, _conf )
    #-------------------------------------------------------------------------
    return _singlet, _multiplet, _Lset

def line_with_text(_ax, _lp, _rp, _text, _tsize, _r, _tangle=0, _lcolor="black", _lwidth=0.7, _lstyle='-', _tcolor="black"):
    r"""
    Parameters
    ----------

    _ax : matlotlib.pyplot.Axe
        the axe to plot a line

    _lp : tuple of float/int, (x,y)
        left point of the line

    _rp : tuple of float/int, (x,y)
        right point of the line

    _text : str
        string of the text

    _tsize : int
        texture size

    _r : float, in the range of [0 : 1]
        relative position along the line starting from left point

    _tangle : int or float
        angle to rotate the text, default : 0

    _lcolor : str
        color of the line, default : "black"

    _lwidth : float
        line width, default : 0.7

    _lstyle : str
        line style, default : '-'

    _tcolor : str
        color of the text, default : "black"

    Returns
    -------

    _line_obj : matplotlib.lines.Line2D
        object of the line we plot

    _text_obj : matplotlib.text.Text
        object of the text we plot
    """
    # {'color': 'black', 'fontsize': 6, 'ha': 'center', 'va': 'center',
    #    'bbox': dict(boxstyle="round", fc="white", ec="white", pad=0.2)}
    _line_obj, = _ax.plot([_lp[0], _rp[0]], [_lp[1], _rp[1]], linestyle=_lstyle, linewidth=_lwidth, color=_lcolor)
    _tx = _lp[0] + (_rp[0]-_lp[0]) * _r
    _ty = _lp[1] + (_rp[1]-_lp[1]) * _r
    _text_obj = _ax.text( _tx, _ty, _text, {'color': _tcolor, 'fontsize': _tsize,
                        'ha': 'center', 'va': 'center','rotation':_tangle,
                        'bbox': dict(boxstyle="round", fc="white", ec="white", pad=0.2)} )

    return _line_obj, _text_obj

class Grotrian:

    def __init__(self, _atom, _conf_prefix):
        r"""

        Parameters
        -----------

        _atom : AtomCls.Atom
            object of the atomic model

        _conf_prefix : str
            common configuration string of the inner shell

        """
        self.atom = _atom

        #---------------------------------------------------------------------
        # prepare structures for plotting
        #---------------------------------------------------------------------
        singlet, multiplet, Lset = _prepare_dict(_atom=_atom, _conf_prefix=_conf_prefix)
        self.singlet = singlet
        self.multiplet = multiplet
        self.Lset = Lset
        #---------------------------------------------------------------------

        #---------------------------------------------------------------------
        # a dictionary to store level position of levels in the plot.
        #---------------------------------------------------------------------
        self.pos_level = {}
        #---------------------------------------------------------------------

    def make_fig(self, _figsize=(6,8), _dpi=120, _f=200):
        r"""

        Parameters
        ----------

        _figsize : tuple of int
            size of the figure, default : (6,8)

        _dpi : int
            "dot per inch" (resolution) of the figure, default : 120

        _f : int
            enlarge factor to show fine structure explicitly, default : 200
        """

        Lset = self.Lset
        singlet = self.singlet
        multiplet = self.multiplet
        pos_level = self.pos_level

        #---------------------------------------------------------------------
        # config
        #---------------------------------------------------------------------
        _hw = 0.5                       # half width of a level in the plot
        _b = len(Lset["singlet"])       # bias in the x axis of multiplet
        _fontsize = 14                  # fontsize of labels
        _textsize = 10                  # fontsize of text of term
        _Jsize = 7                      # fontsize of text of J
        _st = 0.1                       # horizontal spacing between term and text
        #---------------------------------------------------------------------

        fig = plt.figure(figsize=_figsize, dpi=_dpi)
        self.fig = fig

        #---------------------------------------------------------------------
        # singlet
        #---------------------------------------------------------------------
        for k0, v0 in singlet.items():
            for k1, v1 in v0.items():
                #plot level
                xs_level = v1[1]-_hw, v1[1]+_hw
                ys_level = v1[0], v1[0]
                plt.plot(xs_level, ys_level, "-k", linewidth=1)
                # store level posiiton, (conf_origin, term, J)
                pos_level[ (v1[2], k0[1], k1) ] = {}
                pos_level[ (v1[2], k0[1], k1) ]["xs"] = xs_level
                pos_level[ (v1[2], k0[1], k1) ]["ys"] = ys_level
                # plot text
                x_text = xs_level[1] + _st
                y_text = ys_level[0]
                plt.text(x_text, y_text, "{} {}".format(k0[0],k0[1]), fontsize=_textsize, color="k")

        # a vertical line separates singlet panel and multiplet panel
        plt.axvline(x=_b+1, linestyle="--", linewidth=0.5, color="k")
        #---------------------------------------------------------------------

        #---------------------------------------------------------------------
        # plotting multiplet
        #---------------------------------------------------------------------
        _hw = 0.6                    # multiplet need wider
        _sf = 0.1                    # space for connecting fine structure
        for k0, v0 in multiplet.items():
            # compute mean term energy
            y_count = 0.
            y_mean = 0
            for k1, v1 in v0.items():
                y_mean += v1[0]
                y_count += 1
            y_mean /= y_count

            for k1, v1 in v0.items():
                # plot level
                y_pos = y_mean + _f * (v1[0]-y_mean)            # enlarge space
                ys_level = y_pos, y_pos
                xs_level = v1[1]+1-_hw+_b, v1[1]+1+_hw+_b-_sf
                plt.plot(xs_level, ys_level, "-k", linewidth=1)
                # store level posiiton, (conf_origin, term, J)
                pos_level[ (v1[2], k0[1], k1) ] = {}
                pos_level[ (v1[2], k0[1], k1) ]["xs"] = xs_level
                pos_level[ (v1[2], k0[1], k1) ]["ys"] = ys_level
                # connect fine structure
                plt.plot([xs_level[1],xs_level[1]+_sf], [y_pos,y_mean], "--k", linewidth=0.5)
                # plot text of J
                plt.text(xs_level[0]-2*_st, y_pos, k1, fontsize=_Jsize, color="k")
            # plot text of term
            x_pos_text = xs_level[1]+_sf + _st
            plt.text(x_pos_text, y_mean, "{} {}".format(k0[0],k0[1]), fontsize=_textsize, color="k")
        #---------------------------------------------------------------------

        #---------------------------------------------------------------------
        # config of the plot
        #---------------------------------------------------------------------

        #--- tick of singlet
        xtick1 = list(Lset["singlet"])
        xticklabel1 = [Cst.L_i2s[xt] for xt in xtick1]
        #--- tick of multiplet
        xtick2 = [ x+_b+1 for x in list(Lset["multiplet"]) ]
        xticklabel2 = [Cst.L_i2s[xt-_b-1] for xt in xtick2]
        #--- customize xticks and xticklabels
        plt.xticks( xtick1 + xtick2, xticklabel1 + xticklabel2, fontsize=_fontsize )

        #--- x,y label; title
        plt.xlabel("L", fontsize=_fontsize)
        plt.ylabel("E [eV]", fontsize=_fontsize, rotation=90)
        plt.title(self.atom.Title, fontsize=_fontsize, y=1)

        #--- change x limit
        plt.xlim(-1, xtick2[-1]+2)
        #---------------------------------------------------------------------

        ylim = self.fig.gca().get_ylim()
        xlim = self.fig.gca().get_xlim()
        # ylim_max - ylim_min
        self.yr = ylim[1] - ylim[0]
        # xlim_max - xlim_min
        self.xr = xlim[1] - xlim[0]

    def save_fig(self, _filename, _dpi=120):
        r"""

        Parameters
        ----------

        _filename : str
            filename (including path) to save your figure

        _dpi : int
            "dot per inch" (resolution) to save figure, default : 120s
        """

        self.fig.savefig(_filename, dpi=_dpi)

    def show_fig(self):

        plt.show()

    def connect_line(self, _cfj1, _cfj2, _r1, _r2, _c, _text, _tsize=7, _r=0.4, _lcolor="black", _lwidth=0.7, _lstyle='-', _tcolor="black"):
        r"""

        Parameters
        ----------

        _cfj1 : tuple of str, (conf,term,J)
            to specify level 1

        _cfj2 : tuple of str, (conf,term,J)
            to specify level 2

        _r1 : float, in the range of [0 : 1]
            relative position along the line of level 1 starting from left point

        _r2 : float, in the range of [0 : 1]
            relative position along the line of level 2 starting from left point

        _c : str
            color of the line you're going to draw

        _text : str
            content of text

        _tsize : int
            fontsize of text, default : 7

        _r : float, in the range of [0 : 1]
            relative position along the line starting from left point, default : 0.4

        _lcolor : str
            color of the line, default : "black"

        _lwidth : float
            line width, default : 0.7

        _lstyle : str
            line style, default : '-'

        _tcolor : str
            color of the text, default : "black"
        """

        pos_lvl1 = self.pos_level[_cfj1]
        pos_lvl2 = self.pos_level[_cfj2]

        # initialize left/right point of the line
        _lp = pos_lvl1["xs"][0] + (pos_lvl1["xs"][1] - pos_lvl1["xs"][0])*_r1 , pos_lvl1["ys"][0]
        _rp = pos_lvl2["xs"][0] + (pos_lvl2["xs"][1] - pos_lvl2["xs"][0])*_r2 , pos_lvl2["ys"][0]
        # if the initialization is not correct, swap them
        if _lp[0] > _rp[0] :
            _lp, _rp = _rp, _lp

        # compute the angle to rotate text
        _w, _h = self.fig.get_size_inches()
        _dx = (_rp[0]-_lp[0]) / self.xr * _w
        _dy = (_rp[1]-_lp[1]) / self.yr * _h
        _tangle = math.atan2( _dy, _dx )
        _tangle = math.degrees( _tangle )
        # plot line and text
        _ax = self.fig.gca()
        _line_obj, _text_obj = line_with_text(_ax=_ax, _lp=_lp, _rp=_rp, _text=_text, _tsize=_tsize, _r=0.5,
                            _tangle=_tangle, _lcolor=_lcolor, _lwidth=_lwidth, _lstyle=_lstyle, _tcolor=_tcolor)
