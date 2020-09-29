
import os
import glob


def _check_output(output):

    status = False
    if "Your branch is up to date" in output:
        status = True

    return status



if __name__ == "__main__":

    #-- check pwd

    _pwd = os.getcwd()
    _idx = _pwd.find("spectra")

    _root_path = _pwd[:_idx+7]
    _module_path = _root_path + "/external/modules/"
    _files = glob.glob(_module_path+'/*')
    _folder_path_list = [_f for _f in _files if os.path.isdir(os.path.join(_f))]

    _uptodate = []
    _needupdate = []
    for _folder_path in _folder_path_list:
        _folder = _folder_path.split('/')[-1]
        os.chdir(_folder_path)
        _output = os.popen(f"git status").read()
        if _check_output(output=_output):
            _uptodate.append(_folder)
        else:
            _needupdate.append(_folder)


    print("\nUp to date :")
    for _folder in _uptodate:
        print(f"  {_folder}")
    print("\nNeed update : ")
    for _folder in _needupdate:
        print(f"  {_folder}")
    print('\n')
