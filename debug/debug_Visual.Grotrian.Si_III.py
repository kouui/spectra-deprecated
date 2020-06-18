

if __name__ == "__main__":

    import sys
    sys.path.append("..")

    from src.Structure import AtomCls
    from src.Visual import Grotrian
    #-------------------------------------------------------------------------
    # C III Be like
    #-------------------------------------------------------------------------
    file = "../data/atom/Si_III/Si_III.Level"

    atom = AtomCls.Atom(file)
    gro = Grotrian.Grotrian(_atom=atom, _conf_prefix="1s2.2s2.2p6.")
    gro.make_fig(_figsize=(10,6),_f=50)

    gro.show_fig()
    #gro.save_fig("../grotrian_diagram/" + filename.split('/')[-1].replace(".txt",".png"))
