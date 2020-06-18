

if __name__ == "__main__":

    import sys
    sys.path.append("..")

    from src.Structure import AtomCls
    from src.Visual import Grotrian
    #-------------------------------------------------------------------------
    # C III Be like
    #-------------------------------------------------------------------------
    file = "../data/atom/C_III/C_III.Level"
    file_Aji = "../data/atom/C_III/Einstein_A/Nist.Aji"
    file_CEe = "../data/atom/C_III/Collisional_Excitation/Berrington_et_al_1985.Electron"
    atom = AtomCls.Atom(file, _file_Aji=file_Aji, _file_CEe=file_CEe)
    gro = Grotrian.Grotrian(_atom=atom, _conf_prefix="1s2.")
    gro.make_fig(_figsize=(10,6))

    line_plot = (
        (0, 4, "977", 0.3, 0.5),
        (0, 2, "1909", 0.7, 0.1),
        (4, 9, "1247", 0.3, 0.5),
        (4, 8, "2297", 0.8, 0.5),
        (2, 5, "1175.99", 0.1, 0.1),
        (1, 6, "1175.26", 0.25, 0.25),
        (2, 6, "1175.59", 0.40, 0.40),
        (3, 6, "1176.38", 0.55, 0.55),
        (2, 7, "1174.93", 0.70, 0.70),
        (3, 7, "1175.71", 0.85, 0.85),
    )
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    # O V Be like
    #-------------------------------------------------------------------------
    ##file = "../data/atom/O_V/O_V.Level"
    ##file_Aji = "../data/atom/O_V/Einstein_A/Nist.Aji"
    ##file_CEe = "../data/atom/O_V/Collisional_Excitation/Berrington_et_al_1985.Electron"
    ##atom = AtomCls.Atom(file, _file_Aji=file_Aji, _file_CEe=file_CEe)
    ##gro = Grotrian.Grotrian(_atom=atom, _conf_prefix="1s2.")
    ##gro.make_fig(_figsize=(10,6))
    ##
    ##line_plot = (
    ##    (0, 4, "629.7", 0.3, 0.5),
    ##    (0, 2, "1218.3", 0.7, 0.1),
    ##    (4, 9, "774.5", 0.3, 0.5),
    ##    (4, 8, "1371.3", 0.8, 0.5),
    ##    (2, 5, "761.1", 0.1, 0.1),
    ##    (1, 6, "759.4", 0.25, 0.25),
    ##    (2, 6, "760.2", 0.40, 0.40),
    ##    (3, 6, "762.0", 0.55, 0.55),
    ##    (2, 7, "758.7", 0.70, 0.70),
    ##    (3, 7, "760.4", 0.85, 0.85),
    ##)
    #-------------------------------------------------------------------------
    for i, j, wl, _r1, _r2 in line_plot:
        _ctj1 = atom.Level_info_table[i]
        _ctj2 = atom.Level_info_table[j]
        gro.connect_line(_cfj1=_ctj1, _cfj2=_ctj2, _r1=_r1, _r2=_r2, _c="black", _text=wl, _tsize=7, _r=0.4)
    gro.show_fig()
    #gro.save_fig("../grotrian_diagram/" + filename.split('/')[-1].replace(".txt",".png"))
