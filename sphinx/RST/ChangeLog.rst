Change Log
======================

2020/07/15
----------------

1. in `src.Uilt.Class`, we added functions

  - `help` : print all attributes and methods in the class object

  - `help_method` : print all methods in the class object

  - `help_attribute` : print all attributes in the class object

2. in `src.Structure.AtomCls`, in class `Atom()`, attribute `.am` has been renamed to `.Mass`


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
