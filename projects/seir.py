import datetime
import time

import pandas as pd
from timeit import default_timer as timer
from datetime import timedelta
import numpy as np
from typing import List
from numpy import double
from logic.compartments import Compartments
from logic.disease_model import DiseaseModel
from logic.utils import Utils
from root import DIR_OUTPUT


class SEIR(DiseaseModel):
    """ Class used to represent an Contact Tracing disease model """

    def __init__(self, _compartments: List[Compartments], r0: float):
        """
        Initialize the run of contact tracing disease model
        """
        super().__init__(_compartments, r0=r0)
        self.population = Utils.population(file='population', year=2020)
        self._num_comp = len(_compartments)
        self.util = Utils()

    def equations(self, x, t, **kwargs):
        try:
            dx = np.zeros(self._num_comp, dtype=double)
            b = kwargs.get('b') if type(kwargs.get('b')) is float else 1.0
            beta = kwargs.get('beta') if type(kwargs.get('beta')) is float else 1.0
            gamma = kwargs.get('gamma') if type(kwargs.get('gamma')) is float else 1.0
            delta = kwargs.get('delta') if type(kwargs.get('delta')) is float else 1.0
            s, e, i, r = x
            n = s + e + i + r
            ds_dt = {1: (beta * s * i) / n, 2: b * n}
            de_dt = {1: (beta * s * i) / n, 2: gamma * e}
            di_dt = {1: gamma * e, 2: delta * i}
            dr_dt = {1: delta * i}

            dx[0] = -ds_dt[1] + ds_dt[2]
            dx[1] = de_dt[1] - de_dt[2]
            dx[2] = di_dt[1] - di_dt[2]
            dx[3] = dr_dt[1]
            return dx
        except Exception as e:
            print('Error equations: {0}'.format(e))
            return None


if __name__ == "__main__":
    start_processing_s = time.process_time()
    start_time = datetime.datetime.now()
    susc = Compartments(name="susceptible", value=999999)
    expo = Compartments(name="exposed", value=0)
    inf = Compartments(name="infectious", value=1)
    rec = Compartments(name="recovered", value=0)
    dead = Compartments(name="dead", value=0)

    compartments = [susc, expo, inf, rec]
    ct = SEIR(_compartments=compartments, r0=0.0)
    kwargs = {'b': 0.012/365, 'beta': 0.4, 'gamma': 1/5, 'delta': 1/5}
    resp = ct.run(days=365, **kwargs)
    file_csv = DIR_OUTPUT + "{0}.csv".format('seir_base_v6')
    df = pd.DataFrame.from_dict(resp)
    df.to_csv(file_csv)
    end_processing_s = time.process_time()
    end_processing_ns = time.process_time_ns
    end_time = datetime.datetime.now()
    print(df.to_string())
    print('Performance: {0}'.format(end_processing_s - start_processing_s))
    time_diff = (end_time - start_time)
    execution_time = time_diff.total_seconds() * 1000
    print('Execution Time: {0} milliseconds'.format(execution_time))
