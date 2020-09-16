from abc import ABC
from typing import List
from pandas import np
from scipy.integrate import odeint
from logic.compartments import Compartments
from logic.transitions import Transitions


class DiseaseModel(ABC):

    def __init__(self, compartments: List[Compartments], value_a: float, value_b: float, value_c: float):
        """
            Init Disease Model with parameters.

        Parameters
        ----------
            compartments : str
                name of compartment.
            transitions : list
                List of transitions compartment between compartment.
        Returns
        -------
            object
                object disease model.
        """
        self._compartments = compartments
        self.value_a = value_a
        self.value_b = value_b
        self.value_c = value_c

    def equations(self, x, t, r0):
        """
            Time derivative of the state vector.
        Parameters
        ----------
            x : object
                the state vector (array_like)
            t : int
                time (scalar)
            r0 : float
                the effective transmission rate, defaulting to a constant
        Returns
        -------
            object
                disease model equations.
        """
        pass

    def solve(self, r0, t_vec, x_init: list):
        """
            Solve for i(t) and c(t) via numerical integration, given the time path for R0..
        Parameters
        ----------
            r0 : double
                rate force of infection
            t_vec : int
                time (scalar)
            x_init : list
                list of initial values each compartment
        Returns
        -------
            dict
                disease model equations.
        """
        try:
            result = odeint(lambda x, t: self.equations(x, t, r0), x_init, t_vec)
            result = np.transpose(result)
            for k, _ in enumerate(result):
                self._compartments[k].result = result[k]
                self._compartments[k].value = x_init[k]
            return {state.name: state.result for state in self._compartments}
        except Exception as e:
            print('Error solve: {0}'.format(e))
            return None

    def result(self, days: int, r0: float):
        """
            Returns all values of the disease model.
        Parameters
        ----------
            arg : dict
                name and values of initial compartments
            days : int
                days of calculate
            r0 : float
                the effective transmission rate, defaulting to a constant
        Returns
        -------
            dict
                values by compartment.
        """
        try:
            val = [c.value for c in self._compartments]
            t_vec = np.linspace(start=0, stop=days, num=days)
            resp = self.solve(r0, t_vec, val)
            return resp
        except Exception as e:
            print('Error set_model: {0}'.format(e))
            return None

