"""
This module defines several class for Solar Atmosphere model.
"""

import numpy as np

from .. import Element as Elem


################################################################################
# parent class of Cartesian atmosphere
################################################################################

class AtmosphereCartesian:
    r"""
    parent class of Cartesion Atmosphere classes.

    **Don't use this class since it is designed as an abstract parent class.**

    Attributes
    ----------

    dType : numpy.dtype
        declare data type of all physical parameters in atmosphere

        - Te : temparature, np.double, [:math:`K`]

        - Vt : turbulent velocity, np.double, [:math:`cm \cdot s^{-1}`]

        - Ne : electron density, np.double, [:math:`cm^{-3}`]

        - Pg : gas pressure, np.double, [:math:`Ba = g \cdot cm^{-1} \cdot s^{-2}`]

        - Vd : line of sight Doppler velocity, np.double, [:math:`cm \cdot s^{-1}`]

        - Mesh : geometry mesh ranging from 0 to 1, np.double, [-]

        - Spatial : geometry mesh of physical scale, np.double, [:math:`cm`]

    abun : array-like, dtype of np.double
        abundance of each element relative to hydrogen

    BackRad : numpy.2darray, dtype of np.double
        axis 0 is wavelength in [:math:`cm`],
        and axis 1 is intensity in [:math:`erg/cm^2/Sr/cm/s`]

    """
    def __init__(self):
        r"""
        decleare attributes `dType` and `abun`
        """

        #: data type of numpy.recarray to store data
        self.dType = np.dtype([
            ('Te',np.double),                       #: electron temperature
            ('Vt',np.double),                       #: turbulent velocity
            ('Ne',np.double),                       #: electron density
            ('Pg',np.double),                       #: gas pressure
            ('Vd',np.double),                       #: doppler velocity
            ('Mesh',np.double),                     #: spatial mesh [0,1]
            ('Spatial',np.double)                   #: spatial mesh [0,thickness]
        ])

        #: currently, we adapt that abundance is spatial independent --> uniform abundance
        self.abun = (10.**(np.array(Elem.AbunTuple)-12.) ).astype(np.double)

    def __str__(self):

        return None

    def __repr__(self):

        return None

    def readBackgroundIntensity(self,path):
        r"""
        read background intensity from path to `self.BackRad`

        Parameters
        -----------
        path : str
            path of background radiation intensity data file

        Notes
        -------
        Executing this method create an Attribute called `self.BackRad`,
        which is numpy.2darray, dtype of np.double,

        - axis 0 : wavelength mesh in [:math:`cm`],
        - axis 1 : the intensity in [:math:`erg/cm^2/Sr/cm/s`]
        """
        #: [cm] V.S. [erg/s/cm^2/cm/sr]
        BackRad = np.loadtxt(path, dtype=np.double, skiprows=2)
        self.BackRad = BackRad.T.copy()
        self.BackRad[1,:] /= 3

    @property
    def shape(self):
        return self.data.Mesh.shape

    @property
    def ndim(self):
        return self.data.Mesh.ndim


################################################################################
# zero dimensional Cartesion atmosphere
################################################################################
class AtmosphereC0d(AtmosphereCartesian):
    r"""
    zero dimensional Cartesian Atmosphere,
    inheritting parent class `AtmosphereCartesian`

    Attributes
    ----------
    data : numpy.recarray, dtype of dTpye
        a structured array to store physical parameter

    dType : numpy.dtype
        a data type object decleared in parent class `AtmosphereCartesian`
    """
    def __init__(self, Te=1E4, Vt=5.E5, Ne=1E10, Pg=5E-1, Vd=0.E5):
        r"""

        initialization method of AtmosphereC0d

        Parameters
        ----------
        Te : np.double
            temparature, [:math:`K`]
        Vt : np.double
            turbulent velocity, [:math:`cm \cdot s^{-1}`]
        Ne : np.double
            electron density, [:math:`cm^{-3}`]
        Pg : np.double
            gas pressure, [:math:`Ba = g \cdot cm^{-1} \cdot s^{-2}`]
        Vd : np.double
            line of sight Doppler velocity, [:math:`cm \cdot s^{-1}`]
        """
        super().__init__()
        
        self.data = np.recarray(1, dtype=self.dType)
        self.data.Te[:] = Te
        self.data.Vt[:] = Vt
        self.data.Ne[:] = Ne
        self.data.Pg[:] = Pg
        self.data.Vd[:] = Vd
        self.data.Mesh[:] = np.nan
        self.data.Spatial[:] = np.nan


################################################################################
# one dimensional Cartesion atmosphere
################################################################################

class AtmosphereC1d(AtmosphereCartesian):
    r"""
    one dimensional Cartesian Atmosphere,
    inheritting parent class `AtmosphereCartesian`

    Attributes
    ----------
    Scale : tuple of float or int

        a tuple of size 1, specifying the grometry scale along line of sight

    data : numpy.recarray, dtype of dTpye
        a structured array to store physical parameter

    dType : numpy.dtype
        a data type object decleared in parent class `AtmosphereCartesian`
    """
    def __init__(self, thickness, isHalf=False, Mesh=None):
        r"""

        initialization method.

        Parameters
        ----------
        thickness : np.double

            thickness of your model, [:math:`cm`]

        Mesh : array-like, dtype of np.double, optional

            default : None

            - None : construct a  91 points slab Mesh or half slab Mesh automatically
            - array-like : user-defined mesh grid, with scale from Mesh[0] = 0 to Mesh[1] = 1

        isHalf : boolean, optional

            default : True

            - False : full slab mesh grid, frequent at lower/upper surface, sparse at the middle, 91 grid points.
            - True : upper half of the full slab mesh, 46 grid points.
        """
        super().__init__()
        self.isHalf = isHalf

        #: a tuple to store spatial scale
        self.Scale = (thickness,)

        if Mesh is None :
            Mesh = self.__defaultSlabSpatialMesh__(isHalf)
        else:
            assert Mesh.ndim==1, "keyword argument Mesh must be a one dimensional numpy.ndarray."
            assert Mesh[0]==0 and Mesh[-1]==1, "Mesh runs from 0 to 1 ."

        data = np.recarray(Mesh.shape, dtype=self.dType); self.data = data
        data.Mesh[:] = Mesh[:]
        data.Spatial[:] = data.Mesh[:] * thickness

    def __defaultSlabSpatialMesh__(self, isHalf):
        r"""
        construct slab-type spatial mesh automatically

        Parameters
        ----------
        isHalf : boolean, optional

            default : True

            - False : full slab mesh grid, frequent at lower/upper surface, sparse at the middle, 91 grid points.
            - True : upper half of the full slab mesh, 46 grid points.

        Returns
        -------
        spMesh : array-like, dtype of np.double

            spatial mesh ranging from 0. to 1.
        """

        assert isinstance(isHalf, bool), "keyword argument isHalf must be a boolean variable."

        Mesh = np.empty(91, dtype=np.double)
        Mesh[:46] = np.array([0.0,
                1.E-8, 1.6E-8, 2.5E-8, 4.0E-8, 6.3E-8,
                1.E-7, 1.6E-7, 2.5E-7, 4.0E-7, 6.3E-7,
                1.E-6, 1.6E-6, 2.5E-6, 4.0E-6, 6.3E-6,
                1.E-5, 1.6E-5, 2.5E-5, 4.0E-5, 6.3E-5,
                1.E-4, 1.6E-4, 2.5E-4, 4.0E-4, 6.3E-4,
                1.E-3, 1.6E-3, 2.5E-3, 4.0E-3, 6.3E-3,
                1.E-2, 1.6E-2, 2.5E-2, 4.0E-2, 6.3E-2,
                .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
        if not isHalf:
            Mesh[46:] = 2*Mesh[45]  - Mesh[0:45][::-1]
            Mesh[:] /= 2
            return Mesh
        else :
            return Mesh[:46]

        self.setUniform()

    def setUniform(self, Te=1E4, Vt=5.E5, Ne=1E10, Pg=5E-1, Vd=0.E5):
        r"""

        set all parameters uniform across the space

        Parameters
        ----------
        Te : np.double
            temparature, [:math:`K`]
        Vt : np.double
            turbulent velocity, [:math:`cm \cdot s^{-1}`]
        Ne : np.double
            electron density, [:math:`cm^{-3}`]
        Pg : np.double
            gas pressure, [:math:`Ba = g \cdot cm^{-1} \cdot s^{-2}`]
        Vd : np.double
            line of sight Doppler velocity, [:math:`cm \cdot s^{-1}`]
        """
        self.data.Te[:] = Te
        self.data.Vt[:] = Vt
        self.data.Ne[:] = Ne
        self.data.Pg[:] = Pg
        self.data.Vd[:] = Vd

    def __repr__(self):

        S = "----------------------------------------\n"
        S+= "1D Cartesian Atmosphere \n"
        S+= "----------------------------------------\n"
        S+= "Z      : {0:<.3e} km \n".format(self.Scale[0]*1E-5)
        S+= "#Depth : {0:<3d} \n".format(self.data.size)
        S+= "isHalf : {} \n".format(self.isHalf)
        S+= "----------------------------------------\n"

        return S

    def __str__(self):

        return self.__repr__()

if __name__=='__main__':
    pass
    #pwd = './'
    #atmos = AtmosphereC1d(1)
    #atmos.readBackgroundIntensity(pwd+'data/IntensityData/ASTM_E490+PROM7_HI_CaII.dat')
