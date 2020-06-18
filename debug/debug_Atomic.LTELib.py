
if __name__ == "__main__":

    import sys
    sys.path.append("..")

    from src.Structure import AtomCls
    from src.Atomic import LTELib

    file     = "../data/atom/C_III/C_III.Level"
    file_Aji = "../data/atom/C_III/Einstein_A/Nist.Aji"
    file_CEe = "../data/atom/C_III/Collisional_Excitation/Berrington_et_al_1985.Electron"
    atom = AtomCls.Atom(file, _file_Aji=file_Aji, _file_CEe=file_CEe)
    nRatio = LTELib.get_LTE_ratio(atom.Level.erg[:], atom.Level.g[:], atom.Level.stage[:], 2E+04, 1E+10)

    wave = 6.563E-5 # [cm]
    Te = 1.E+4 # [K]
    print(LTELib.EinsteinA_to_EinsteinBs_cm(0, wave, 2,8))
    print(LTELib.Planck_cm(wave, Te))
