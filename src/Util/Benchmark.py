
import matplotlib.pyplot as plt


def make_figure(x : list, result : dict, save_path : str, title=None, _ax=None):

    if _ax is None:
        fig, ax = plt.subplots(1,1,figsize=(6,4), dpi=100)
    else:
        ax = _ax

    for key in result.keys():
        ax.plot(x, result[key], "--o", markersize=3, label=key)

    ax.legend(loc="upper left")
    ax.set_yscale("log")
    ax.set_xscale("log")

    ylim_upper =  max( [max( li ) for li in result.values()] )
    ylim_upper = max( [ylim_upper, 1E-1] )
    ax.set_ylim(1E-6, ylim_upper)

    if title is not None:
        ax.set_title( title )

    if _ax is None:
        fig.savefig(save_path, dpi=100)


def print_table(x : list, result : dict, outFile=None):


    nbar = 16 + 16*len(result.keys())
    s = '-'*nbar + '\n'
    s += "unit [1E-6 sec] "
    for key in result.keys():
        s += f"| {key:^13s} "
    s += '\n'
    s += '-'*nbar + '\n'

    for k in range(len(x)):
        s += f"n={str(x[k]):<13s} "
        for val in result.values():
            ss = f"{val[k]*1E6:<.1f}"
            s +=  f"| {ss:>9s}     "
        s += '\n'
    s += '-'*nbar + '\n'

    print(s)

    if outFile is not None:
        with open(outFile, 'w') as f:
            f.write( s )
