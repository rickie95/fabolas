import numpy as np
from scipy.special import exp1


class Horseshoe:

    def __init__(self, scale=1):
        self.scale = scale

    def logpdf(self, x: float):
        """
            Compute the log pdf. 
            Relies on E1(x) implementation from Scipy.

            ## Parameters

            - `x`: `float`

            ## Returns 

            - logpdf of `x` : `float` or `-np.inf`
        """
        ln_pi = 1.1447298858494001
        ln_2  = 0.6931471805599453

        term = (x / self.scale)**2 / 2
        logs = (ln_2 + 3 * ln_pi + 2 * np.log(self.scale))
        exp_integral_1 = exp1(term)

        if exp_integral_1 > 0:
            return  term - logs / 2 + np.log(exp_integral_1)
        
        return -np.inf
