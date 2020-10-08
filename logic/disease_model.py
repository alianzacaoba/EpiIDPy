from abc import ABC
from typing import List
from pandas import np
from scipy.integrate import odeint
from logic.compartments import Compartments
from logic.transitions import Transitions


class DiseaseModel(ABC):
    """ Class used to represent an Disease Model"""

    def __init__(self, compartments: List[Compartments], r0: float = 0.0, value_b: float = 0.0, value_c: float = 0.0):
        """ Init Disease Model with parameters.
        :param compartments: name of compartment.
        :type compartments: List
        :returns: Object Disease model
        :rtype: object
        """
        self._compartments = compartments
        self._r0 = r0
        self.value_b = value_b
        self.value_c = value_c

    def equations(self, x, t, **kwargs):
        """Time derivative of the state vector.
        :param x: The compartment vector (array_like)
        :type x: Object Compartments
        :param t: time (scalar)
        :type t: int
        :param r0: The effective transmission rate, defaulting to a constant
        :type r0: float
        :returns: Disease model equations.
        :rtype: object
        """
        pass

    def __solve(self, x_init: list, time_vector, **kwargs):
        """Solve for dx(t) and d(t) via numerical integration, given the time path for R0.
        :param x_init: List of initial values each compartment (float list)
        :type x_init: List
        :param time_vector:  time (scalar)
        :type time_vector: int
        :param r0: rate force of infection
        :type r0: double
        :returns: Disease model equations.
        :rtype: dict
        """
        try:
            result = odeint(lambda x, t: self.equations(x, t, **kwargs), x_init, time_vector)
            result = np.transpose(result)
            for k, _ in enumerate(result):
                self._compartments[k].result = result[k]
                self._compartments[k].value = x_init[k]
            return {state.name: state.result for state in self._compartments}
        except Exception as e:
            print('Error solve: {0}'.format(e))
            return None

    def run(self, days: int, **kwargs):
        """Returns all values of the disease model.
        :param days: days of calculate
        :type days: int
        :param r0: the effective transmission rate, defaulting to a constant
        :type r0: double
        :returns: Values by compartment.
        :rtype: dict
        """
        try:
            compartments_values = [c.value for c in self._compartments]
            time_vector = np.linspace(start=0, stop=days, num=days)
            resp = self.__solve(x_init=compartments_values, time_vector=time_vector, **kwargs)
            return resp
        except Exception as e:
            print('Error set_model: {0}'.format(e))
            return None

