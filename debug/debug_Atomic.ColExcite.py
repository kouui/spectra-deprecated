
if __name__ == "__main__":

    import sys
    sys.path.append("..")

    import numpy as np
    from src.Structure import AtomCls
    from src.Atomic import LTELib, ColExcite

    file     = "../data/atom/C_III/C_III.Level"
    file_Aji = "../data/atom/C_III/Einstein_A/Nist.Aji"
    file_CEe = "../data/atom/C_III/Collisional_Excitation/Berrington_et_al_1985.Electron"
    atom = AtomCls.Atom(file, _file_Aji=file_Aji, _file_CEe=file_CEe)
    Te = 2E+04
    ne = 1E+10

    #--- compute LTE population ratio for each CE transition
    n_LTE = LTELib.get_LTE_ratio(_erg=atom.Level.erg[:], _g=atom.Level.g[:],
                    _stage=atom.Level.stage[:], _Te=Te, _Ne=ne)
    nTran = atom.CE_table.shape[0]
    ni_LTE = np.empty(nTran, np.double)
    nj_LTE = np.empty(nTran, np.double)

    for k in range(nTran):
        ni_LTE[k] = n_LTE[atom.CE_coe.idxI[k]]
        nj_LTE[k] = n_LTE[atom.CE_coe.idxJ[k]]
    #---
    CE_fac = ColExcite.interpolate_CE_fac(_table=atom.CE_table[:,:], _Te=Te, _Te_table=atom.CE_Te_table[:],
                            _f1=atom.CE_coe.f1[:], _f2=atom.CE_coe.f2[:])
    CEij = ColExcite.get_CE_rate_coe(_CE_fac=CE_fac, _Te=Te, _gi=atom.CE_coe.gi[:],
                            _dEij=atom.CE_coe.dEij[:], _type=atom.CE_type)
    CEji = ColExcite.Cij_to_Cji(_Cij=CEij,  _ni_LTE=ni_LTE, _nj_LTE=nj_LTE)


    # test scipy B-spline interpolation
    ##import matplotlib.pyplot as plt
    ##from scipy.interpolate import splrep, splev
    ##k = 1
    ##Bsp_obj = splrep(x=atom.CE_Te_table[:], y=atom.CE_table[k,:])
    ##Te_arr = np.logspace(3,6, 31, endpoint=True)
    ##res_arr = splev(Te_arr, Bsp_obj, ext=3)
    ##fig = plt.figure(figsize=(8,5), dpi=80)
    ##plt.plot( atom.CE_Te_table[:], atom.CE_table[k,:], "*")
    ##plt.plot( Te_arr[:], res_arr[:], "-")
    ##plt.show()
