import time
import numpy as np
from typing import List
from numpy import double
from logic.compartments import Compartments
from logic.disease_model import DiseaseModel
from logic.transitions import Transitions
from logic.settings import arg, DAYS, GAMMA, age_groups, R0
# import graphviz


class ConctactTracing(DiseaseModel):
    """
    Simulate continuous-time SIRPy epidemics.
    Object Oriented Bayesian Network for SIR Models
    All units in the simulator are in hours for numerical stability, though disease parameters are
    assumed to be in units of days as usual in epidemiology
    """

    def __init__(self, _compartments: List[Compartments], _transitions: List[Transitions],
                 _beta: float, _gamma: float):
        """
        Initialize the run of the epidemic
        State and queue codes (transition event into this state)
        """
        super().__init__(compartments=_compartments, transitions=_transitions, beta=_beta, gamma=_gamma)
        self._num_comp = len(_compartments)

    def solve(self, r0, t_vec, x_init: list):
        return super(ConctactTracing, self).solve(r0, t_vec, x_init)

    def result(self, arg: dict, days: int, r0):
        return super(ConctactTracing, self).result(arg, days, r0)
    
    def equations(self, x, t, r0):
        try:
            dx = np.zeros(self._num_comp, dtype=double)
            s, e, i, r = x
            beta = r0(t) * self._beta if callable(r0) else r0 * self._beta
            ne = beta * s * i
            # Time derivatives

            dx[0] = -ne
            dx[1] = ne - self._beta * e
            dx[2] = beta * e - self._gamma * i
            dx[3] = self._gamma * i
            return dx
        except Exception as e:
            print('Error equations: {0}'.format(e))
            return None


if __name__ == "__main__":
    start_time = time.time()
    susc = Compartments(name='susceptible')
    expo = Compartments(name='exposed')
    iinf = Compartments(name='infectious_symptoms')
    rec = Compartments(name='recovered')
    dead = Compartments(name='dead')
    # susc_expo = Transitions(rate=0.2, probability=0.5, org=susc, dest=expo).value

    compartments = [susc, expo, iinf, rec]
    transitions = []

    result = []
    for key, value in age_groups.items():
        dict_temp = {key: value}
        ct = ConctactTracing(_compartments=compartments, _transitions=transitions, _gamma=GAMMA, _beta=value)
        resp = ct.result(arg=arg, days=DAYS, r0=R0)
        dict_temp.update({item['_name']: item['_result'] for item in resp})
        result.append(dict_temp)

    print(result)
    # Calculated Time processing
    t_sec = round(time.time() - start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    time_processing = '{} hour:{} min:{} sec'.format(t_hour, t_min, t_sec)
    print('Time processing: {0}'.format(time_processing))