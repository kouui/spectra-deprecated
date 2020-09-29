Change Log 2020/06~
============================

2020/09/29
----------

1. updated homepage

2. git branch :

  - local : master

  - remote : origin/master

2020/08/18
-----------

1. folder in `*.conf` is changed to relative path

2. in `src/Function/StatisticalEquilibrium/LibArray.py`, added function

  - `convert_ni_to_nj_by_ni`

3. modified `rate` in `notebooks/StatisticalEquilibrium/SE_Plots.ipynb`

4. completed `notebooks/StatisticalEquilibrium/SE_Plots.ipynb`

  - net rate

  - markdown equations



2020/08/10
-----------

1. added visualization module `src/Visual/DragLine2D` for interactive 1D radiative transfer

2020/08/08
-----------

1. modified `src.Atomic.PhotoIonize.bound_free_radiative_transition_coefficient` to have correct `Rki_spon` (in wavelength unit)

2. in `src/Visual/Grotrian`, we renamed

  - `Grotrian.show_transition_rate()` -> `Grotrian.plot_transition_rate()`

  - `Grotrian.show_Grotrian()` -> `Grotrian.plot_transitions()`

3. modified `gj/gi` to `float(gi)/float(gi)` since they give different accuracy because `gi` and `gj` are integers

4. moved `notebooks/StatisticalEquilibrium/Lib.py` to `src/Function/StatisticalEquilibrium/LibArray.py`

5. moved `notebooks/StatisticalEquilibrium/Lib.2py` to `src/Function/StatisticalEquilibrium/LibClass.py`

6. in `src/Structure/AtomCls`, we added function `InitAtom()`, which takes `/path/to/*.conf` file as an input and returns a `Atom` object

2020/08/07
---------------

1. now the configuration string in data files are without prefix

2. in `src/Visual/Grotrian`, we added

  - `Grotrian.show_transition_rate()`

  - `Grotrian.show_Grotrian()`

3. `Grotrian` now has configuration data file like `*.Grotrian`

4. in `src/Structure/AtmosphereCls`, we added

  - `AtmosphereCartesian.read_BackRad_QS()`

  - `AtmosphereCartesian.BackRad_int`

  - `AtmosphereCartesian.BackRad_wav`

  - `AtmosphereCartesian.BackRadAir_int`

  - `AtmosphereCartesian.BackRadAir_wav`

5. added `TODO.rst`


2020/07/30
---------------

0. 1-year schedule in `notebooks/StatisticalEquilibrium/Plots_v2.ipynb`

1. in `data/atom/Ca_II/Ca_II.Level`, energies have been adjusted back to NIST values

  - previous problem is caused by wavelength in vacuum/air

2. in `src.Atomic.BasicP`, added functions

  - `refractive_index_in_air`

  - `air_to_vacuum`

3. now "3000-10000A" and "10000-13400A" is corresponded to wavelength in vacuum.

4. added `notebooks/StatisticalEquilibrium/Lib2.py` for the sake of high level function calling

  - the usage of `notebooks/StatisticalEquilibrium/Lib2.py` is illustrated in `notebooks/StatisticalEquilibrium/Plots_v2.ipynb`




2020/07/20
---------------

1. `src.Uilt.Class.help()` now can response to both `class` and `recarray`

2. in `data/atom/Ca_II/Ca_II.Level`, energies have been adjusted to fit each central wavelength

3. quiet sun intensity atlas is now available : `data/intensity/atlas/QS/atlas_QS.npy`

    - `notebooks/Atlas/Atlas_QS.ipynb` gives a naive illustration to this atlas


2020/07/15
----------------

1. in `src.Uilt.Class`, we added functions

  - `help` : print all attributes and methods in the class object

  - `help_method` : print all methods in the class object

  - `help_attribute` : print all attributes in the class object

2. in `src.Structure.AtomCls`, in class `Atom()`, attribute `.am` has been renamed to `.Mass`

3. reconstructed `src.Structure.AtomCls.Atom()`, now we have

  - `Atom().Level`, `Atom().Line`, `Atom.Cont` to store information of energy level, line transition and continuum transitions, respectively.

  - `src.Structure.AtomCls.Collisional_Transition()` to store information of Collisional Excitation/Ionization

  - `src.Structure.AtomCls.Photoionization()` to store information of Photoionization

  - `src.Structure.MeshCls.WavelengthMesh()` to take charge of wavelength construction for Line/Continuum

  - `src.Structure.RadLineCls.RadiativeLine()` to take charge of radiative lines


2020/06/26
-----------------

1. in `src.Structure.AtmosphereCls`, we added `AtmosphereC0d`

2. dependency of numba has been changed from 0.43.0 to 0.49.1, and this will not affect code execution.

3. in `src.Structure.AtomCls`, `Atom.PI_table` has been changed to `Atom.PI_table_list`. all the list attributes in `Atom()` are created by

  .. code-block::

    ... = = []

  however, in the future, this will be changed to

  .. code-block::

    from numba.typed import List
    ... = List()

4. module `src.Atomic.PhotoIonize` has been created for functions related to photoionization/recombination.

5. in `src.Atomic.PhotoIonize`, we added

  - `interpolate_PI_intensity`

  - `interpolate_PI_alpha`

  - `bound_free_radiative_transition_coefficient`

6. in `src.Atomic.BasicP`, we added

  - `get_Level_gamma`

  - `get_Line_Gamma`

  - `get_damping_a`

7. in `src.RadiativeTransfer.Profile`, we added

  - `half_to_full`

8. `src.Atomic.ColExcite` has been reconstructed to `src.Atomic.Collision`, including functions for both collisional (de)excitation and collisional ionization/recombination. Functions includes

  - `interpolate_CE_fac`, input argument changed

  - `interpolate_CI_fac`

  - `Cij_to_Cji`

  - `get_CE_rate_coe`

  - `get_CI_rate_coe`

9. `notebooks/Demo_FullStatisticalEquilirbium_v1.ipynb` created. Demo of full statistical equilibrium at single depth using Ca_II is completed(?).

10. `Attention.rst` has been added to the main page of documentation website. Currently, it includes

  - `Convention of wavelength-frequency conversion` : related to wavelength/frequency mesh. intensity unit and input argument of Voigt function.


2020/06/23
-----------------

1. added documentation `./index.html`
