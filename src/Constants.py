################################################################################
#This module defines physics constants and
#some other constants for the convenience of computation.
#
#Modification History:
#    2020.5.8   k.i.   alp_
#
################################################################################

################################################################################
# mathematical constants
################################################################################

pi_ = 3.141592653589793                 #: mathematical constant pi, [-]

################################################################################
# constants for unit conversion
################################################################################

eV2erg_ = 1.60217662 * 1.E-12           #: unit conversion from eV to erg, [:math:`erg/eV`]
J2erg_ = 1.E+7                          #: unit conversion from J to erg, [:math:`erg/J`]
m2cm_ = 1.E+2                           #: unit conversion from m to cm, [:math:`cm/m`]
micro2AA_ = 1.E4                        #: unit conversion from micro to angstrom, [:math:`A / \mu m`]
K2eV_ = 8.6173324 * 1.E-5               #: compute electron temperature in unit eV, [:math:`eV / K`]
eV2AA_div_ = 1.23984176 * 1E+4          #: unit conversion from eV to angstrom, [:math:`eV \cdot A`]

################################################################################
# physcis constants
################################################################################

c_ = 2.9979246  * 1.E+10                #: speed of light, [:math:`cm \cdot s^{-1}`]
h_ = 6.62606885 * 1.E-27                #: Planck constant, [:math:`erg \cdot s`]
k_ = 1.3806485  * 1.E-16                #: Boltzmann constant, [:math:`erg \cdot k^{-1}`]
e_ = 4.80320425 * 1.E-10                #: eleceton charge, [:math:`esu`]
mH_ = 1.660     * 1.E-24                #: mass of hydrogen atom, [:math:`g`]
me_ = 9.1093836 * 1.E-28                #: mass of electron, [:math:`g`]
E_Rydberg_ = 2.1798741 * 1.E-11         #: Rydberg constant of hydrogen atom, [:math:`erg`]
a0_ = 5.2917720859 * 1.E-9              #: Bohr radius, [:math:`cm`]
alp_ = 1/137.036                        #: fine structure constant, [:math:`-`]
AU_ = 1.49597871 * 1.E+13               #: astronomical unit, distance from the sun to our earth, [:math:`cm`]
R_sun_ = 6.957 * 1.E+10                 #: Solar radius, [:math:`cm`]

################################################################################
# constants for the convenience of computation
################################################################################

sqrtPi_ = 1.7724538509055159            #: square root of pi, [-]
sqrt3_ = 1.73205080757                  #: sqrt(3), [-]

saha_ = 2 * (2*pi_*me_*k_/h_/h_)**(1.5)
"""a constant factor in Saha's equation, :math:`2(2 \pi m_e k /h^2)^{3/2}`, [:math:`g^{3/2} \cdot erg^{-3/2} \cdot k^{-3/2} \cdot s^{-3}`]
"""

################################################################################
# constants for notations
################################################################################

L_s2i = { "S" : 0, "P" : 1, "D" : 2, "F" : 3, "G" : 4, "H" : 5, "I" : 6 }
"""a hash dictionary mapping symbolic quantum number L to its integer value
"""
L_i2s = { 0 : "S", 1 : "P", 2 : "D", 3 : "F", 4 : "G", 5 : "H", 6 : "I" }
"""a hash dictionary mapping integer value to its symbolic quantum number L
"""

################################################################################
# constants for code configuration
################################################################################
isJIT = True
"""whether to turn on the JIT compilation in all *.py files,
since sphinx does no understand numba thus can not generate documentation for numba jitted functions.

set to
    True  : before simulation; before pushing to github

    False : before generating documentation using sphinx
"""
