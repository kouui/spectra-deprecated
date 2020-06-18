"""
This module defines chemical elements as global variables.
"""


#-- element table
ElementTuple = ('He','C','N','O','Ne','Na','Mg','Al','Si','P',
            'S','Ar','K','Ca','Cr','Mn','Fe','Co','Ni','H')

#-- atomic number
nZTuple = (2, 6, 7, 8, 10, 11, 12, 13, 14, 15,
        16, 18, 19, 20, 24, 25, 26, 27, 28, 1)

#-- relative atom mass, where 'C' has value of 12
AMTuple = (4.003, 12.01, 14.01, 16.00, 20.18,
        23.00, 24.32, 26.97, 28.09, 30.97,
        32.07, 39.94, 39.10, 40.08, 52.01,
        54.93, 55.85, 58.94, 58.69, 1.008)

#--  relative abundance
AbunTuple = (11.00, 8.54, 8.06, 8.83, 7.55,
            6.45, 7.54, 6.45, 7.65, 5.45,
            7.21, 6.75, 5.70, 6.40, 6.00,
            5.55, 7.72, 5.35, 6.40, 12.00)
