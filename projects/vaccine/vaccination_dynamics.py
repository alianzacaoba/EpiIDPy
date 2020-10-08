from datetime import time
import numpy
from typing import List
from numpy import double
from pandas import np
from logic.compartments import Compartments
from logic.disease_model import DiseaseModel
from logic.utils import Utils
from config.vaccine_setting import rate


class VaccinationDynamics(DiseaseModel):
    """Class used to represent an Vaccine Cost Effectiveness"""

    def __init__(self, _compartments: List[Compartments], r0: float, value_b: float, value_c: float):
        super().__init__(_compartments, r0, value_b, value_c)
        self._num_comp = len(_compartments)
        self.population = Utils.load_population(file='population', year=2020)
        self.total_matrix = Utils.load_contact_matrices(file='total_contact_matrix')
        self.vaccine_capacities = Utils.region_capacities(file='region_capacities')
        self.priority_vaccine = Utils.priority(file='priority')

    def __vaccine_assignment(self, age_group: List, vaccine_capacities: int, priority_vaccine: list) -> float:
        try:
            size_population = sum(age_group)
            vaccine_assignment = np.zeros(size_population, dtype=double)
            total_candidate = vaccine_capacities
            for pv in priority_vaccine:
                vaccine_assignment[pv] = min(age_group[pv], total_candidate)
                total_candidate -= vaccine_assignment[pv]

            return vaccine_assignment
        except Exception as e:
            print('Error vaccine_assignment: {0}'.format(e))
            return 0.0

    def new_comparment(self, contact_matrix: numpy.matrix, age_group: list):
        print('VAsig')

    def equations(self, x, t, **kwargs):
        try:
            age_group = kwargs.get('age_groups') if type(kwargs.get('age_groups')) is list else list()
            vaccine_capacities = kwargs.get('vaccine_capacities') if type(kwargs.get('vaccine_capacities')) is int else 1
            priority_vaccine = kwargs.get('priority_vaccine') if type(kwargs.get('priority_vaccine')) is list else list()
            dx = np.zeros(self._num_comp, dtype=double)
            su, f_1, f_2, e, a, a_f, r_a, v_1, v_2 = x
            va_sig = self.__vaccine_assignment(age_group=age_group,
                                               vaccine_capacities=vaccine_capacities,
                                               priority_vaccine=priority_vaccine)

            ds_dt = su * va_sig
            f1_dt = {1: -f_1 * va_sig,
                     2: (1 - rate['EPSILON_1']) * su * va_sig}
            f2_dt = (1 - rate['EPSILON_2']) * f_1 * va_sig
            de_dt = -e * va_sig
            da_dt = -a * va_sig
            daf_dt = (1 - rate['PS']) * e * va_sig
            dra_dt = -r_a * va_sig
            v1_dt = {1: v_1 * va_sig,
                     2: rate['EPSILON_1'] * su * va_sig,
                     3: a * va_sig,
                     4: r_a * va_sig}
            v2_dt = {1: v_1 * va_sig,
                     2: rate['EPSILON_2'] * f_1 * va_sig}

            dx[0] = ds_dt
            dx[1] = f1_dt[1] + f1_dt[2]
            dx[2] = f2_dt
            dx[3] = de_dt
            dx[4] = da_dt
            dx[5] = daf_dt
            dx[6] = dra_dt
            dx[7] = v1_dt[1] + v1_dt[2] + v1_dt[3] + v1_dt[4]
            dx[8] = v2_dt[1] + v2_dt[2]
            return dx
        except Exception as e:
            print('Error equations: {0}'.format(e))
            return None


if __name__ == "__main__":
    compartments = []
    sus = Compartments(name='Susceptible', value=0.0)
    compartments.append(sus)
    first_dose = Compartments(name='Population_1Dose', value=0)
    compartments.append(first_dose)
    second_dose = Compartments(name='Population_2Dose', value=0)
    compartments.append(second_dose)
    expo = Compartments(name='Exposed', value=10)
    compartments.append(expo)
    asym = Compartments(name='Asymptomatic', value=0)
    compartments.append(asym)
    asym_f = Compartments(name='Asymptomatic_F', value=0)
    compartments.append(asym_f)
    ra = Compartments(name='Recovered ', value=0)
    compartments.append(ra)
    v1 = Compartments(name='Vaccinated_1 ', value=0)
    compartments.append(v1)
    v2 = Compartments(name='Vaccinated_2 ', value=0)
    compartments.append(v2)
    ct = VaccinationDynamics(_compartments=compartments, r0=3.0, value_b=0.0, value_c=0.0)
    result = {}
    setting = {'matrix': ct.total_matrix}
    for dept, age_groups in ct.population.items():
        setting.update({'age_groups': list(dict(age_groups).values()),
                        'vaccine_capacities': ct.vaccine_capacities[dept],
                        'priority': ct.priority_vaccine})
        resp = ct.run(days=100, **setting)

    #Utils.save_json('vaccine_cost_effectiveness', result)
    #Utils.save_scv('vaccine_cost_effectiveness', result)
    print(result)





