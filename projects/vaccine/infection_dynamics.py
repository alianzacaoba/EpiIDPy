from datetime import time
import numpy
from typing import List
from numpy import double
from pandas import np
from logic.compartments import Compartments
from logic.disease_model import DiseaseModel
from logic.utils import Utils
from config.vaccine_setting import rate


class InfectionDynamics(DiseaseModel):
    """Class used to represent an Vaccine Cost Effectiveness"""

    def __init__(self, _compartments: List[Compartments], r0: float, value_b: float, value_c: float):
        super().__init__(_compartments, r0, value_b, value_c)
        self._num_comp = len(_compartments)
        self.population = Utils.load_population(file='population', year=2020)
        self.total_matrix = Utils.load_contact_matrices(file='total_contact_matrix')

    def new_comparment(self, contact_matrix: numpy.matrix, age_group: list):
        print('VAsig')

    def equations(self, x, t, **setting):
        try:
            age_group = kwargs.get('age_group') if type(kwargs.get('age_group')) is list else list()
            dx = np.zeros(self._num_comp, dtype=double)
            su, f_1, f_2, e, a, a_f, r_a, v_1, v_2 = x

            dx[0] = 1

            return dx
        except Exception as e:
            print('Error equations: {0}'.format(e))
            return None


if __name__ == "__main__":
    compartments = []
    sus = Compartments(name='Susceptible', value=0.0)
    compartments.append(sus)
    first_dose = Compartments(name='Population_first_dose', value=0)
    compartments.append(first_dose)
    second_dose = Compartments(name='Population_second_dose', value=0)
    compartments.append(second_dose)
    expo = Compartments(name='Population_symptoms', value=10)
    compartments.append(expo)
    pre = Compartments(name='Presymptoms', value=0)
    compartments.append(pre)
    ct = InfectionDynamics(_compartments=compartments, r0=3.0, value_b=0.0, value_c=0.0)
    result = {}
    kwargs = {'matrix': ct.total_matrix}
    for dept, age_groups in ct.population.items():
        kwargs.update({'age_groups': dict(age_groups).values()})
        resp = ct.run(days=100, **kwargs)

    #Utils.save_json('vaccine_cost_effectiveness', result)
    #Utils.save_scv('vaccine_cost_effectiveness', result)
    print(result)





