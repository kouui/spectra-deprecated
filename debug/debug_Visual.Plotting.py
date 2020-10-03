
import sys
sys.path.append( '../' )

from src.Visual import Plotting

import matplotlib.pyplot as plt



if __name__ == "__main__":

    img = [[1,2],[3,4]]

    fig_kw = {
        "figsize" : (8,4),
        "dpi" : 100,
    }

    axe_kw = {
        "ax1" : [0,0,0.5,1],
        "ax2" : [0.5,0,0.5,1]
    }

    fig, axe = Plotting.axes_no_padding(fig_kw=fig_kw, axe_kw=axe_kw)
    axe["ax1"].imshow(img)
    axe["ax2"].imshow(img)

    plt.show()
