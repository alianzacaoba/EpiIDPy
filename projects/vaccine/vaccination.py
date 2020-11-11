import numpy
import pandas
from typing import List
from numpy import double
from logic.compartments import Compartments
from logic.disease_model import DiseaseModel


class Vaccination(DiseaseModel):
    """Class used to represent an Vaccine Cost Effectiveness"""

    def __init__(self, _compartments: List[Compartments], r0: float):
        super().__init__(_compartments, r0)
        self._num_comp = len(_compartments)

    def __vaccine_assignment(self, candidates: dict, vaccine_capacities: int, priority_vaccine: dict) -> dict:
        # Candidates: su, v1,f1,e,a,ra
        try:
            vaccine_assigment = dict()
            remaining = vaccine_capacities
            for pv in priority_vaccine:
                age_group, work_group, risk = pv.values()  # Age group, Workgroup, Health risk
                vac_age_group = {age_group: dict()} if age_group not in vaccine_assigment else vaccine_assigment[age_group]
                vac_age_group[work_group] = {work_group: dict()} if work_group not in vac_age_group else vac_age_group[work_group]
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
            beta = kwargs.get('beta') if type(kwargs.get('beta')) is float else 0.0
            total_population = kwargs.get('total_population') if type(kwargs.get('total_population')) is float else 1.0
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
            age_group = kwargs.get('age_group') if type(kwargs.get('age_group')) is str else str
            health_group = kwargs.get('health_group') if type(kwargs.get('health_group')) is str else str
            work_group = kwargs.get('work_group') if type(kwargs.get('work_group')) is str else str
            calibration = kwargs.get('calibration') if type(kwargs.get('calibration')) is float else 0.0
            arrival_rate = kwargs.get('arrival_rate') if type(kwargs.get('arrival_rate')) is float else 0.0
            population_initial = kwargs.get('population_initial') if type(
                kwargs.get('population_initial')) is dict else dict()
            vaccine_capacities = kwargs.get('vaccine_capacities') if type(
                kwargs.get('vaccine_capacities')) is int else 1
            priority_vaccine = kwargs.get('priority_vaccine') if type(
                kwargs.get('priority_vaccine')) is list else list()
            contact_matrix = kwargs.get('contact_matrix') if type(
                kwargs.get('contact_matrix')) is dict else dict()
            dx = numpy.zeros(self._num_comp, dtype=double)
            su, f_1, f_2, e, e_f, a, a_f, p, sy, c, h, i, r, r_a, v_1, v_2, d = x
            sum_x = sum([su, f_1, f_2, e, a, a_f, r_a, v_1, v_2])
            # i_1 = [a, a_f, p, sy]
            # i_2 = [c, h, i]
            va_sig = self.__vaccine_assignment(candidates=population_initial,
                                               vaccine_capacities=vaccine_capacities,
                                               priority_vaccine=priority_vaccine)[0] if calibration else 0.0

            i_1 = list()
            for kg, vg in population_initial.items():
                total = 0
                for kw, vw in dict(vg).items():
                    for kh, vh in dict(vw).items():
                        value = double(str(vh).strip())
                        total += value
                i_1.append(total)

            i_2 = list()
            for kg, vg in population_initial.items():
                total = 0
                for kw, vw in dict(vg).items():
                    if kw == 'M':
                        for kh, vh in dict(vw).items():
                            value = double(str(vh).strip())
                            total += value
                i_2.append(total)

            contact_i1 = numpy.array(contact_matrix[age_group]) * numpy.array(i_1)
            contact_i2 = numpy.array(contact_matrix[age_group]) * numpy.array(i_2)
            if work_group == 'M':
                prod = sum(contact_i1) + sum(contact_i2)
            else:
                prod = sum(contact_i1)
            # ----------------------------------------------------------------------------------------------------------
            ds_dt = su * va_sig
            f1_dt = {1: -f_1 * va_sig,
                     2: (1 - epsilon_1) * su * va_sig}
            f2_dt = (1 - epsilon_2) * f_1 * va_sig
            de_dt = {1: -e * va_sig, 2: (sum_x / total_population) * arrival_rate}
            da_dt = -a * va_sig
            daf_dt = (1 - p_s) * e * va_sig
            dra_dt = -r_a * va_sig
            v1_dt = {1: v_1 * va_sig,
                     2: epsilon_1 * su * va_sig,
                     3: a * va_sig,
                     4: r_a * va_sig}
            v2_dt = {1: v_1 * va_sig,
                     2: epsilon_2 * f_1 * va_sig}
            # ----------------------------------------------------------------------------------------------------------
            ds = {1: (-(beta * su) / total_population) * prod}
            df1 = {1: (-(beta * f_1) / total_population) * prod}
            df2 = {1: (-(beta * f_2) / total_population) * prod}
            dee = {1: e / t_e, 2: (-(beta * su) / total_population) * prod}
            df = {1: e_f / t_e, 2: (-(beta * (f_1 + f_2)) / total_population) * prod}

            # ----------------------------------------------------------------------------------------------------------
            da = {1: -a / t_a, 2: (1 - p_s) * (e / t_e)}
            da_f = {1: -a_f / t_a, 2: (1 - p_s) * (e_f / t_e)}
            dp = {1: p_s * ((e + e_f / t_e) + e * va_sig), 2: p / t_p}
            dsy = {1: p / t_p, 2: -sy / t_sy}
            dc = {1: -p_c * (c / t_d), 2: -p_c * (c / t_r), 3: p_c * (sy / t_sy)}
            dh = {1: -p_h * (h / t_d), 2: -p_h * t_r, 3: p_h * (sy / t_sy)}
            di = {1: -p_i * (i / t_d), 2: -p_i * (i / t_r), 3: p_i * (sy / t_sy)}
            draa = {1: a / t_a}
            dr = {1: (p_c * c + p_h * h + p_i * i) / t_r}
            dv2 = {1: a_f / t_a}
            dd = {1: p_c * (c / t_d), 2: p_h * (h / t_d), 3: p_i * (i / t_d)}

            # ----------------------------------------------------------------------------------------------------------
            dx[0] = ds_dt + ds[1]
            dx[1] = sum([vf for kf, vf in f1_dt.items()]) + df1[1]
            dx[2] = f2_dt + df2[1]
            dx[3] = sum([ve for ke, ve in de_dt.items()]) + sum([vee for kee, vee in dee.items()])
            dx[4] = da_dt + sum([va for ka, va in da.items()]) + sum([vaa for kaa, vaa in draa.items()])
            dx[5] = daf_dt + sum([vf for kf, vf in da_f.items()])
            dx[6] = dra_dt
            dx[7] = sum([vv for kv, vv in v1_dt.items()])
            dx[8] = sum([vv for kv, vv in v2_dt.items()]) + sum([vv for kv, vv in dv2.items()])
            # ---------------------------------------------
            dx[9] = sum([vf for kf, vf in df.items()])
            dx[10] = sum([vp for kp, vp in dp.items()])
            dx[11] = sum([vp for kp, vp in dsy.items()])
            dx[12] = sum([vc for kc, vc in dc.items()])
            # ---------------------------------------------
            dx[13] = sum([vh for kh, vh in dh.items()])
            dx[14] = sum([vi for ki, vi in di.items()])
            dx[15] = sum([vr for kr, vr in dr.items()])
            dx[16] = sum([vd for kd, vd in dd.items()])
            return dx
        except Exception as e:
            print('Error equations: {0}'.format(e))
            return None
