from datetime import time
from typing import List
from numpy import double
from pandas import np
from logic.compartments import Compartments
from logic.disease_model import DiseaseModel
from logic.utils import Utils
from config.vaccine_setting import rate, time_comp, work_risk, age_groups, region, N


class VaccineCostEffectiveness(DiseaseModel):
    """Class used to represent an Vaccine Cost Effectiveness"""

    def __init__(self, compartments: List[Compartments], value_a: float, value_b: float, value_c: float):
        super().__init__(compartments, value_a, value_b, value_c)
        self._num_comp = len(compartments)

    def solve(self, x_init: list, time_vector, r0):
        return super(VaccineCostEffectiveness, self).solve(x_init, time_vector, r0)

    def result(self, days: int, r0: float):
        return super(VaccineCostEffectiveness, self).result(days, r0)

    def equations(self, x, t, r0):
        try:
            age_group = self.value_b
            work_risk = self.value_a
            dx = np.zeros(self._num_comp, dtype=double)
            s, f_1, f_2, e, p = x
            #  s, f_1, f_2, e, e_f, a, a_f, p, c, h, i, r, r_a, v_1, v_2, d = x

            ds_dt = rate['BETA'] * (s / age_group) * (p * age_group + p * work_risk)
            f1_dt = -rate['BETA'] * (f_1 / age_group) * (s / age_group) * (p * age_group + p * work_risk)
            f2_dt = -rate['BETA'] * (f_2 / age_group) * (s / age_group) * (p * age_group + p * work_risk)
            de_dt = {1: -e/time_comp['t_e'],
                     2: -rate['BETA'] * (s / age_group) * (f_2 / age_group) * (s / age_group) * (p * age_group + p * work_risk)}
            def_dt = {1: -e/time_comp['t_e'],
                      2: -rate['BETA'] * (f1_dt + f2_dt / age_group) * (s / age_group) * (p * age_group + p * work_risk)}

            dx[0] = ds_dt
            dx[1] = f1_dt
            dx[2] = f2_dt
            dx[3] = de_dt[1] + de_dt[2]
            dx[4] = def_dt[1] + def_dt[2]
            return dx
        except Exception as e:
            print('Error equations: {0}'.format(e))
            return None


if __name__ == "__main__":
    compartments = []
    sus = Compartments(name='susceptible', value=N*0.8)
    compartments.append(sus)
    first_dose = Compartments(name='Population_first_dose', value=0)
    compartments.append(first_dose)
    second_dose = Compartments(name='Population_second_dose', value=0)
    compartments.append(second_dose)
    expo = Compartments(name='population_symptoms', value=10)
    compartments.append(expo)
    pre = Compartments(name='Presymptoms', value=0)
    compartments.append(pre)

    result = {}
    for key_work, value_work in work_risk.items():
        age_dict = {}
        for key_age, value_age in age_groups.items():
            ct = VaccineCostEffectiveness(compartments=compartments,
                                          value_a=value_work,
                                          value_b=value_age,
                                          value_c=0.0)
            resp = ct.result(days=100, r0=2.4)
            age_dict[key_age] = resp
        result[key_work] = {'age_groups': age_dict}
    Utils.save_json('vaccine_cost_effectiveness', result)
    Utils.save_scv('vaccine_cost_effectiveness', result)
    print(result)





