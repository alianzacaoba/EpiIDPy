import time
import numpy as np
from typing import List
from numpy import double
from logic.compartments import Compartments
from logic.disease_model import DiseaseModel
from logic.transitions import Transitions
from config.contact_settings import arg, DAYS, GAMMA, BETA, age_groups, R0
# import graphviz


class ContactTracing(DiseaseModel):
    """
    Simulate continuous-time SIRPy epidemics.
    Object Oriented Bayesian Network for SIR Models
    All units in the simulator are in hours for numerical stability, though disease parameters are
    assumed to be in units of days as usual in epidemiology
    """

    def __init__(self, compartments: List[Compartments], value_a: float, value_b: float, value_c: float):
        """
        Initialize the run of the epidemic
        State and queue codes (transition event into this state)
        """
        super().__init__(compartments, value_a, value_b, value_c)
        self._num_comp = len(compartments)

    def solve(self, r0, t_vec, x_init: list):
        return super(ContactTracing, self).solve(r0, t_vec, x_init)

    def result(self, days, r0):
        return super(ContactTracing, self).result(days, r0)
    
    def equations(self, x, t, r0):
        try:
            dx = np.zeros(self._num_comp, dtype=double)
            s, e, i, r = x
            beta = r0(t) * BETA if callable(r0) else r0 * BETA
            ne = beta * s * i
            # Time derivatives

            dx[0] = -ne
            dx[1] = ne - GAMMA * e
            dx[2] = beta * e - GAMMA * i
            dx[3] = GAMMA * i
            return dx
        except Exception as e:
            print('Error equations: {0}'.format(e))
            return None


if __name__ == "__main__":
    start_time = time.time()
    susc = Compartments(name="susceptible")
    expo = Compartments(name="exposed")
    iinf = Compartments(name="infectious_symptoms")
    rec = Compartments(name="recovered")
    dead = Compartments(name="dead")
    # susc_expo = Transitions(rate=0.2, probability=0.5, org=susc, dest=expo).value

    compartments = [susc, expo, iinf, rec]
    transitions = []

    result = []
    for key, value in age_groups.items():
        dict_temp = {'age_group': key}
        ct = ContactTracing(compartments=compartments, value_a=GAMMA, value_b=value, value_c=0.0)
        resp = ct.result(days=DAYS, r0=R0)
        dict_temp.update({item['_name']: item['_result'] for item in resp})
        result.append(dict_temp)


    print(result)
    # Calculated Time processing
    t_sec = round(time.time() - start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    time_processing = '{} hour:{} min:{} sec'.format(t_hour, t_min, t_sec)
    print('Time processing: {0}'.format(time_processing))