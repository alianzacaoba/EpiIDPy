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
        self.total_matrix = Utils.contact_matrices(file='total_contact_matrix')
        self.population = Utils.population(file='population', year=2020)
        self.vaccine_capacities = Utils.region_capacities(file='region_capacities')
        self.priority_vaccine = Utils.priority_vaccine(file='priority', scenario=1)

    def __vaccine_assignment(self, candidates: list, vaccine_capacities: int, priority_vaccine: dict) -> dict:
        # Candidates: su, v1,f1,e,a,ra
        try:
            vaccine_assigment = dict()
            remaining = vaccine_capacities
            for pv in priority_vaccine:
                age_group, work_group, risk = pv.values()  # Age group, Workgroup, Health risk
                vac_age_group = vaccine_assigment.get(age_group, dict())
                vac_age_group[work_group] = vac_age_group.get(work_group, dict())
                vac_age_group[work_group][risk] = dict()
                group = candidates[age_group][work_group][risk]
                vac_asig = min(sum(group), remaining)
                for healthGroup in group.keys():
                    vac_age_group[work_group][risk][healthGroup] = vac_asig / sum(group)
                vaccine_assigment[age_group] = vac_age_group
                remaining -= round(vac_asig)
                if remaining == 0:
                    break
            return vaccine_assigment
        except Exception as e:
            print('Error vaccine_assignment: {0}'.format(e))
            return {0: 0.0}

    def equations(self, x, t, **kwargs):
        try:
            vaccine_capacities = kwargs.get('vaccine_capacities') if type(kwargs.get('vaccine_capacities')) is int else 1
            priority_vaccine = kwargs.get('priority_vaccine') if type(
                kwargs.get('priority_vaccine')) is list else list
            dx = np.zeros(self._num_comp, dtype=double)
            su, f_1, f_2, e, a, a_f, r_a, v_1, v_2 = x
            va_sig = self.__vaccine_assignment(candidates=[su, v_1, f_1, e, a, ra],
                                               vaccine_capacities=vaccine_capacities,
                                               priority_vaccine=priority_vaccine)[0]

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
            dx[1] = sum([vf for kf, vf in f1_dt.items()])
            dx[2] = f2_dt
            dx[3] = de_dt
            dx[4] = da_dt
            dx[5] = daf_dt
            dx[6] = dra_dt
            dx[7] = sum([vv for kv, vv in v1_dt.items()])
            dx[8] = sum([vv for kv, vv in v2_dt.items()])
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
        setting.update({'priority_vaccine': ct.priority_vaccine,
                        'vaccine_capacities': ct.vaccine_capacities[dept]})
        resp = ct.run(days=100, **setting)

    # Utils.save_json('vaccine_cost_effectiveness', result)
    # Utils.save_scv('vaccine_cost_effectiveness', result)
    print(result)





