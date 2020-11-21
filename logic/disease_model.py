from abc import ABC
from typing import List
import numpy
from scipy.integrate import odeint
from logic.compartments import Compartments


class DiseaseModel(ABC):
    """ Class used to represent an Disease Model"""

    def __init__(self, compartments: List[Compartments], r0: float = 0.0):
        """ Init Disease Model with parameters.
        :param compartments: name of compartment.
        :type compartments: List
        :returns: Object Disease model
        :rtype: object
        """
        self._compartments = compartments
        self._r0 = r0

    def equations(self, x, t, **kwargs):
        """Time derivative of the state vector.
        :param x: The compartment vector (array_like)
        :type x: Object Compartments
        :param t: time (scalar)
        :type t: int
        :returns: Disease model equations.
        :rtype: object
        """
        pass

    def __solve(self, x_init: list, time_vector: list, **kwargs):
        """Solve for dx(t) and d(t) via numerical integration, given the time path for R0.
        :param x_init: List of initial values each compartment (float list)
        :type x_init: List
        :param time_vector:  time (scalar)
        :type time_vector: int
        :returns: Disease model equations.
        :rtype: dict
        """
        try:
            # http://www.scholarpedia.org/article/Odeint_library
            result = odeint(lambda x, t: self.equations(x, t, **kwargs), x_init, time_vector)
            result = numpy.transpose(result)
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
            time_vector = list(numpy.linspace(start=0, stop=days, num=days, dtype=int))
            resp = self.__solve(x_init=compartments_values, time_vector=time_vector, **kwargs)
            return resp
        except Exception as e:
            print('Error set_model: {0}'.format(e))
            return None

