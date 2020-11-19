import numpy as np
from typing import List, Dict, Any, Union
from numpy import double
from logic.compartments import Compartments
from logic.disease_model import DiseaseModel


class Vaccination(DiseaseModel):
    """Class used to represent an Vaccine Cost Effectiveness"""

    def __init__(self, _compartments: List[Compartments], r0: float, **inf_percent: dict):
        super().__init__(_compartments, r0)
        self._num_comp = len(_compartments)
        self.inf_percent = inf_percent

    def __vaccine_assignment(self, candidates: dict, vaccine_capacities: int, priority_vaccine: dict) -> dict:
        # Candidates: su, v1,f1,e,a,ra
        try:
            vaccine_assigment = dict()
            remaining = vaccine_capacities
            for pv in priority_vaccine:
                age_group, work_group, risk = pv.values()  # Age group, Workgroup, Health risk
                vac_age_group = {age_group: dict()} if age_group not in vaccine_assigment else vaccine_assigment[
                    age_group]
                vac_age_group[work_group] = {work_group: dict()} if work_group not in vac_age_group else vac_age_group[
                    work_group]
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
            beta = kwargs.get('beta') if type(kwargs.get('beta')) is float else 0.5
            total_population = kwargs.get('total_population') if type(
                kwargs.get('total_population')) is dict else dict()
            initial_population = kwargs.get('initial_population') if type(
                kwargs.get('initial_population')) is dict else dict()
            epsilon_1 = kwargs.get('epsilon_1') if type(kwargs.get('epsilon_1')) is float else 0.0
            epsilon_2 = kwargs.get('epsilon_2') if type(kwargs.get('epsilon_2')) is float else 0.0
            t_e = kwargs.get('t_e') if type(kwargs.get('t_e')) is float else 1.0
            t_a = kwargs.get('t_a') if type(kwargs.get('t_a')) is float else 1.0
            t_p = kwargs.get('t_p') if type(kwargs.get('t_p')) is float else 1.0
            t_sy = kwargs.get('t_sy') if type(kwargs.get('t_sy')) is float else 1.0
            t_r = kwargs.get('t_r') if type(kwargs.get('t_r')) is float else 1.0
            t_d = kwargs.get('t_d') if type(kwargs.get('t_d')) is float else 1.0
            p_s = kwargs.get('p_s') if type(kwargs.get('p_s')) is float else 0.0
            p_c = kwargs.get('p_c') if type(kwargs.get('p_c')) is float else 0.0
            p_h = kwargs.get('p_h') if type(kwargs.get('p_h')) is float else 0.0
            p_i = kwargs.get('p_i') if type(kwargs.get('p_i')) is float else 0.0
            p_dc = kwargs.get('p_dc') if type(kwargs.get('p_dc')) is float else 0.0
            p_dh = kwargs.get('p_dh') if type(kwargs.get('p_dh')) is float else 0.0
            p_di = kwargs.get('p_di') if type(kwargs.get('p_di')) is float else 0.0
            health = kwargs.get('health') if type(kwargs.get('health')) is str else str
            work = kwargs.get('work') if type(kwargs.get('work')) is str else str
            age = kwargs.get('age') if type(kwargs.get('age')) is str else str
            calibration = kwargs.get('calibration') if type(kwargs.get('calibration')) is bool else False
            arrival_rate = kwargs.get('arrival_rate') if type(kwargs.get('arrival_rate')) is float else 0.0
            vaccine_capacities = kwargs.get('vaccine_capacities') if type(
                kwargs.get('vaccine_capacities')) is int else 1
            priority_vaccine = kwargs.get('priority_vaccine') if type(
                kwargs.get('priority_vaccine')) is dict else dict()
            contact_matrix = kwargs.get('contact_matrix') if type(kwargs.get('contact_matrix')) is dict else dict()

            dx = np.zeros(self._num_comp, dtype=double)
            su, f_1, f_2, e, e_f, a, a_f, p, sy, c, h, i, r, r_a, v_1, v_2, d, cases = x
            sum_x = sum([su, f_1, f_2, e, a, a_f, r_a, v_1, v_2])
            va_sig = self.__vaccine_assignment(candidates=initial_population,
                                               vaccine_capacities=vaccine_capacities,
                                               priority_vaccine=priority_vaccine)[0] if not calibration else 0.0

            contagion_states_1 = [a, a_f, p, sy]
            contagion_states_2 = [c, h, i]
            i_1 = float(sum(contagion_states_1))
            i_2 = float(sum(contagion_states_2))

            validate = not self.inf_percent['inf_percent'][age]
            if validate:
                self.inf_percent['inf_percent'][age] = {'value_1': i_1, 'value_1_old': 0.0,
                                                        'value_2': i_2, 'value_2_old': 0.0,
                                                        'total': float(sum_x), 'total_old': 1.0}
            else:
                tmp_dict = self.inf_percent['inf_percent'][age]
                tmp_dict['value_1'] = i_1 + tmp_dict['value_1']
                tmp_dict['value_1_old'] = i_1
                tmp_dict['value_2'] = i_2 + tmp_dict['value_2']
                tmp_dict['value_2_old'] = i_2
                tmp_dict['total'] = float(sum_x) + tmp_dict['total']
                tmp_dict['total_old'] = float(sum_x)
                self.inf_percent['inf_percent'][age] = tmp_dict

            '''
            #su * beta * (1 - np.prod(np.power(1 - inf_percent_1, np.array(contact_matrix[ka]))))

            contact_i1 = numpy.array(contact_matrix[age_group]) * numpy.array(i_1)
            if work_group == 'M':
                i_2 = list()
                for kg, vg in initial_population.items():
                    total = 0
                    for kw, vw in dict(vg).items():
                        for kh, vh in dict(vw).items():
                            if kh in contagion_states_2:
                                value = float(str(vh).strip())
                                total += value
                    i_2.append(total)
                contact_i2 = numpy.array(contact_matrix[age_group]) * numpy.array(i_2)
                prod = sum(contact_i1) + sum(contact_i2)
            else:
                prod = sum(contact_i1)
            '''
            # ----------------------------------------------------------------------------------------------------------
            try:
                contagion_sus = 0.0
                '''
                    (beta * (su * (1 - va_sig) / total_population) * prod) \
                    if ((beta * (su * (1 - va_sig) / total_population) * prod) - su * (1 - va_sig)) * 100000 > 0.0 \
                    else max(su * (1 - va_sig), 0.0)
                '''
                # print('BETA:',beta, 'su', su,  'va_sig', va_sig, 'total_population', total_population, 'prod', prod,
                #      'contagion_sus', contagion_sus)
                contagion_f1 = 0.0  # (beta * f_1 * (1 - va_sig) / total_population) * prod
                contagion_f2 = 0.0  # ((beta * f_2  / total_population) * prod)
                ds_dt = {1: 0.0,  # -su * va_sig,
                         2: -contagion_sus
                         }
                f1_dt = {1: 0.0,  # -f_1 * va_sig,
                         2: 0.0,  # (1 - epsilon_1) * su * va_sig,
                         3: -contagion_f1
                         }
                f2_dt = {1: 0.0,  # (1 - epsilon_2) * f_1 * va_sig,
                         2: -contagion_f2
                         }
                de_dt = {1: 0.0,  # -e * va_sig,
                         2: (sum_x / sum(total_population.values())) * arrival_rate,
                         3: -(e * (1 - va_sig)) / t_e,
                         4: contagion_sus
                         }
                def_dt = {1: 0.0,  # -e_f / t_e,
                          2: contagion_f1 + contagion_f2
                          }
                da_dt = {1: 0.0,  # -a * va_sig,
                         2: -a * (1 - va_sig) / t_a,
                         3: (1 - p_s) * e * (1 - va_sig) / t_e
                         }
                daf_dt = {1: 0.0,  # (1 - p_s) * e * va_sig,
                          2: 0.0,  # -a_f / t_a,
                          3: 0.0  # (1 - p_s) * (e_f / t_e)
                          }
                dp_dt = {1: p_s * e * (1 - va_sig) / t_e,
                         2: 0.0,  # p_s * e_f / t_e,
                         3: 0.0,  # p_s * e * va_sig,
                         4: -p / t_p
                         }
                dsy_dt = {1: p / t_p,
                          2: -sy / t_sy
                          }
                dc_dt = {1: -p_dc * (c / t_d),
                         2: -(1 - p_dc) * (c / t_r),
                         3: p_c * (sy / t_sy)
                         }
                dh_dt = {1: -p_dh * (h / t_d),
                         2: -(1 - p_dh) * h / t_r,
                         3: p_h * (sy / t_sy)
                         }
                di_dt = {1: -p_di * (i / t_d),
                         2: -(1 - p_di) * (i / t_r),
                         3: p_i * (sy / t_sy)
                         }
                dr_dt = {1: ((1 - p_dc) * c + (1 - p_dh) * h + (1 - p_di) * i) / t_r
                         }
                dra_dt = {1: 0.0,  # -r_a * va_sig,
                          2: a * (1 - va_sig) / t_a
                          }
                v1_dt = {1: 0.0,  # -v_1 * va_sig,
                         2: 0.0,  # epsilon_1 * su * va_sig,
                         3: 0.0,  # a * va_sig,
                         4: 0.0  # r_a * va_sig
                         }
                v2_dt = {1: 0.0,  # v_1 * va_sig,
                         2: 0.0,  # epsilon_2 * f_1 * va_sig,
                         3: 0.0,  # a_f / t_a
                         }
                dd_dt = {1: p_dc * (c / t_d),
                         2: p_dh * (h / t_d),
                         3: p_di * (i / t_d)
                         }
                # ----------------------------------------------------------------------------------------------------------
                dx[0] = sum([vs for ks, vs in ds_dt.items()])  # SU
                dx[1] = sum([vf for kf, vf in f1_dt.items()])  # F1
                dx[2] = sum([vf for kf, vf in f2_dt.items()])  # F2
                dx[3] = sum([ve for ke, ve in de_dt.items()])  # E
                dx[4] = sum([vef for kef, vef in def_dt.items()])  # EF
                dx[5] = sum([va for ka, va in da_dt.items()])  # A
                dx[6] = sum([vaf for kaf, vaf in daf_dt.items()])  # AF
                dx[7] = sum([vp for kp, vp in dp_dt.items()])  # P
                dx[8] = sum([vp for kp, vp in dsy_dt.items()])  # SY
                dx[9] = sum([vc for kc, vc in dc_dt.items()])  # C
                dx[10] = sum([vh for kh, vh in dh_dt.items()])  # H
                dx[11] = sum([vi for ki, vi in di_dt.items()])  # I
                dx[12] = sum([vr for kr, vr in dr_dt.items()])  # R
                dx[13] = sum([vr for kr, vr in dra_dt.items()])  # RA
                dx[14] = sum([vv for kv, vv in v1_dt.items()])  # V1
                dx[15] = sum([vv for kv, vv in v2_dt.items()])  # V2
                dx[16] = sum([vd for kd, vd in dd_dt.items()])  # D
                dx[17] = dp_dt[1]  # Cases
                return dx
            except Exception as e:
                print('Error calculation equations elements: {0}'.format(e))
        except Exception as e:
            print('Error equations: {0}'.format(e))
            return None
