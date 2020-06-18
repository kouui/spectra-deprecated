
if __name__ == "__main__":

    import sys
    sys.path.append("..")

    from src.Structure import AtomCls

    file = "../data/atom/C_III/C_III.Level"
    file_Aji = "../data/atom/C_III/Einstein_A/Nist.Aji"
    file_CEe = "../data/atom/C_III/Collisional_Excitation/Berrington_et_al_1985.Electron"
    atom = AtomCls.Atom(file, _file_Aji=file_Aji, _file_CEe=file_CEe)

    #--- assert that the index - configuration search method works well
    line_index = 2

    line_ctj = atom.line_index_to_line_ctj(line_index)
    line_idx = atom.line_ctj_to_line_idx(line_ctj)
    line_index2 = atom.line_idx_to_line_index(line_idx)

    line_idx = atom.line_index_to_line_idx(line_index)
    line_ctj = atom.line_idx_to_line_ctj(line_idx)
    line_index3 = atom.line_ctj_to_line_index(line_ctj)

    assert line_index == line_index2 == line_index3
