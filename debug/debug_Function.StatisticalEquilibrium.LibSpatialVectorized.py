
import sys
sys.path.append("../")

from src.Structure import AtomCls
from src.Function.StatisticalEquilibrium import LibSpatialVectorized as SELib

from src.Atomic import Collision

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from src.Visual import Plotting

import numpy

if __name__ == "__main__":

    atom_, _ = AtomCls.InitAtom("../data/conf/H.conf", isHydrogen=True)

    #Te_ = 1E4
    #Ne_ = 1E11
    #arr_ = numpy.ones((10,10),dtype=numpy.double)
    #Te_ = arr_ * Te_
    #Ne_ = arr_ * Ne_
    Te0_   = numpy.linspace(5E3, 2E4, 61)
    Ne0_   = numpy.logspace(8,14, 61)
    Te_, Ne_ = numpy.meshgrid(Te0_, Ne0_)
    Tr_ = 6E3

    eta = {}

    n_LTE_ , nj_by_ni_Line_, nj_by_ni_Cont_ = SELib.ni_nj_LTE(atom_, Te_, Ne_)
    Rik_, Rki_stim_, Rki_spon_ = SELib.bf_R_rate(atom_, Te_, nj_by_ni_Cont_, _Tr=Tr_)
    Bij_Jbar_, Bji_Jbar_ = SELib.B_Jbar(atom_, Te_, _Tr=Tr_)
    Cij_ = SELib.get_Cij(atom_, Te_)
    Cji_ = Collision.Cij_to_Cji( Cij_, numpy.append(nj_by_ni_Line_,nj_by_ni_Cont_,axis=0) )
    n_SE_ = SELib.solve_SE(atom_, Ne_, Cji_, Cij_,
                          Bji_Jbar_, Bij_Jbar_,
                          Rki_spon_, Rki_stim_, Rik_)

    eta["Local Termal Equil."] = n_LTE_[-1,:] / n_LTE_[:,:].sum(axis=0)
    eta["Statistical Equil."]  = n_SE_[-1,:] / n_SE_[:,:].sum(axis=0)

    ##-- Coronal Equilibrium
    Bij_Jbar_[:] = 0.
    Bji_Jbar_[:] = 0.
    Rki_stim_[:] = 0.
    Rik_[:]      = 0.
    n_SE_ = SELib.solve_SE(atom_, Ne_, Cji_, Cij_,
                          Bji_Jbar_, Bij_Jbar_,
                          Rki_spon_, Rki_stim_, Rik_)

    eta["Coronal Equil."]  = n_SE_[-1,:] / n_SE_[:,:].sum(axis=0)

    ##-- plot
    fig, axs = plt.subplots(1,3, figsize=(9,3), dpi=150)
    plt.subplots_adjust(wspace=0.03)

    #norm = LogNorm(vmin=1.E-9, vmax=1.)
    norm = Normalize(vmin=1.E-5, vmax=1.)
    for ax, name in zip( axs, eta.keys() ):
        im = ax.imshow(eta[name][:,:], cmap="inferno", origin="lower", norm=norm)

        Plotting.set_imshow_ticks(ax, Te0_, axis='x',points=4, fmt='%1.0f',rot=0)
        Plotting.set_imshow_ticks(ax, numpy.log10(Ne0_), axis='y',points=4, fmt='%1.0f',rot=0)
        ax.set_title("$ n_{H\, II} \; / \; n_{H} $" + f"  {name}")
        ax.grid(True)

    axs[0].set_ylabel("$N_{e} \; [cm^{-3}]$")
    axs[0].set_xlabel("$T_{e} \; [K]$")
    Plotting.remove_tick_ticklabel(axs[1],axs[2], kind='xy')

    cax = fig.add_axes([0.45,0.07,0.4,0.02])
    fig.colorbar(im, cax=cax, orientation="horizontal")

    plt.show()
