
import sys, os
sys.path.append( "../../" )

import scipy.special as Spectial_scipy
from src.Math import Special

if __name__ == "__main__":

    x = 0.1
    error = Special.E1(x) - Spectial_scipy.expn(1, x)
    relative_error = abs( error ) / Spectial_scipy.expn(1, x)

    print(f"E1 : the relative error when x={x:.2f} is {relative_error:1.2E}")

    x = 0.5
    error = Special.E2(x) - Spectial_scipy.expn(2, x)
    relative_error = abs( error ) / Spectial_scipy.expn(2, x)

    print(f"E2 : the relative error when x={x:.2f} is {relative_error:1.2E}")
