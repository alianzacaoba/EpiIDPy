import numpy as np
from typing import List
from numpy import double
from logic.compartments import Compartments
from logic.disease_model import DiseaseModel
from logic.transitions import Transitions
from config.economic_settings import DAYS, prob, rate, N, Mm


class EconomicReactivation(DiseaseModel):
    """Class used to represent an Economic Reactivation"""
    def __init__(self, _compartments: List[Compartments], value_a: float, value_b: float, value_c: float):
        """
        Initialize the run of the epidemic
        State and queue codes (transition event into this state)
        """
        super().__init__(_compartments, value_a, value_b, value_c)
        self._num_comp = len(_compartments)

    def solve(self, r0, t_vec, x_init: list):
        return super(EconomicReactivation, self).solve(r0, t_vec, x_init)

    def result(self, days: int, r0):
        return super(EconomicReactivation, self).result(days, r0)

    def equations(self, x, t, r0):
        try:
            dx = np.zeros(self._num_comp, dtype=double)
            s_oa, e_oa, a_oa, p_oa, i_oa, c_oa, h_oa, u_oa, d_oa, r_oa, \
            s_qoa, e_qoa, a_qoa, p_qoa, i_qoa, r_qoa, A, P, I, Ii, H, U = x
            # Time derivatives
            f = rate['BETA'] * (Mm * (I + P + A) / N)
            q_oa = s_qoa + e_qoa + a_qoa + p_qoa + r_qoa

            # Derivative elements
            ds_qoa = {1: rate['PHI'] * s_qoa, 2: -rate['MU'] * s_qoa, 3: prob['ma'] * s_qoa,
                      4: prob['xoa'] * s_qoa, 5: -prob['koa'] * s_qoa}

            ds_oa = {1: rate['PHI'] * s_oa, 2: -f * s_oa, 3: -rate['MU'] * s_oa,
                     4: prob['ma'] * s_oa, 5: -prob['xoa'] * s_oa, 6: prob['koa'] * s_oa}

            de_qoa = {1: -rate['ALPHA'] * e_qoa, 2: -rate['MU'] * e_qoa, 3: prob['ma'] * e_qoa,
                      4: prob['xoa'] * e_qoa, 5: -prob['koa'] * e_qoa}

            de_oa = {1: f * s_oa, 2: -rate['ALPHA'] * e_oa, 3: -rate['MU'] * e_oa, 4: prob['ma'] * e_oa,
                     5: -prob['hoa'] * e_oa, 6: -prob['ma'] * e_qoa, 7: -prob['xoa'] * e_oa, 8: -prob['koa'] * e_oa}

            da_qoa = {1: rate['ALPHA'] * (1 - prob['ga']) * e_qoa, 2: -rate['THETA'] * a_qoa, 3: -rate['MU'] * a_qoa,
                      4: prob['ma'] * a_qoa, 5: prob['xoa'] * a_oa, 6: -prob['koa'] * a_oa}

            da_oa = {1: rate['ALPHA'] * (1 - prob['ga']) * e_oa, 2: -rate['THETA'] * a_oa, 3: -rate['MU'] * a_oa,
                     4: prob['ma'] * a_oa, 5: -prob['xoa'] * a_oa, 6: prob['koa'] * a_oa}

            dp_qoa = {1: rate['ALPHA'] * prob['ga'] * e_qoa, 2: -rate['DELTA'] * p_qoa, 3: -rate['MU'] * p_qoa,
                      4: prob['ma'] * p_qoa, 5: prob['xoa'] * p_oa, 6: -prob['koa'] * e_oa}

            dp_oa = {1: rate['ALPHA'] * prob['ga'] * e_oa, 2: -rate['DELTA'] * p_oa, 3: -rate['MU'] * p_oa,
                     4: prob['ma'] * p_oa, 5: -prob['xoa'] * p_oa, 6: prob['koa'] * e_oa}

            di_qoa = {1: rate['ALPHA'] * rate['RHO'] * p_oa, 2: rate['ALPHA'] * p_qoa,
                      3: -rate['PHI'] * Ii, 4: prob['ma'] * i_oa}

            di_oa = {1: rate['ALPHA'] * (1 - rate['RHO']) * p_oa, 2: -rate['PHI'] * I, 3: prob['ma'] * i_oa}

            dc_oa = {1: rate['PHI'] * (1 - (prob['ua'] + prob['ha'])) * (I + Ii), 2: -rate['SIGMA'] * c_oa}

            dh_oa = {1: rate['PHI'] * prob['ha'] * (I + Ii), 2: -rate['TAU'] * h_oa}

            du_oa = {1: rate['PHI'] * prob['ua'] * (I + Ii), 2: -rate['OMEGA'] * u_oa}

            dd_oa = {1: rate['SIGMA'] * prob['uac'] * c_oa, 2: rate['TAU'] * prob['uah'] * H,
                     3: rate['OMEGA'] * prob['uau'] * u_oa}

            dr_qoa = {1: rate['THETA'] * a_qoa, 2: prob['ma'] * r_qoa,
                      3: prob['xoa'] * r_qoa, 4: prob['koa'] * r_qoa}

            dr_oa = {1: rate['SIGMA'] * (1 - prob['uac']) * c_oa, 2: rate['TAU'] * (1 - prob['uah']) * h_oa,
                     3: rate['OMEGA'] * (1 - prob['uau']) * u_oa, 4: prob['ma'] * r_oa, 5: rate['THETA'] * a_oa,
                     6: -prob['xoa'] * r_oa, 7: prob['koa'] * r_oa}

            dx[0] = sum(ds_qoa.values())
            dx[1] = sum(ds_oa.values())
            dx[2] = sum(de_qoa.values())
            dx[3] = sum(de_oa.values())
            dx[4] = sum(da_qoa.values())
            dx[5] = sum(da_oa.values())
            dx[6] = sum(dp_qoa.values())
            dx[7] = sum(dp_oa.values())
            dx[8] = sum(di_qoa.values())
            dx[9] = sum(di_oa.values())
            dx[10] = sum(dc_oa.values())
            dx[11] = sum(dh_oa.values())
            dx[12] = sum(du_oa.values())
            dx[13] = sum(dd_oa.values())
            dx[14] = sum(dr_qoa.values())
            dx[15] = sum(dr_oa.values())
            return dx
        except Exception as e:
            print('Error equations: {0}'.format(e))
            return None


if __name__ == "__main__":
    compartments = []
    susc_oa = Compartments(name='susceptible_oa')
    compartments.append(susc_oa)
    expo_oa = Compartments(name='exposed_oa')
    compartments.append(expo_oa)
    asym_oa = Compartments(name='asymptomatic_oa')
    compartments.append(asym_oa)
    pre_oa = Compartments(name='pre-symptomatic_oa')
    compartments.append(pre_oa)
    inf_oa = Compartments(name='infectious_oa')
    compartments.append(inf_oa)
    cih_oa = Compartments(name='homecare_isolated_population_oa')
    compartments.append(cih_oa)
    hosp_oa = Compartments(name='hospitalized_oa')
    compartments.append(hosp_oa)
    u_oa = Compartments(name='isolated_critical_care_oa')
    compartments.append(u_oa)
    dea_oa = Compartments(name='deaths_oa')
    compartments.append(dea_oa)
    rec_oa = Compartments(name='recovered_oa')
    compartments.append(rec_oa)

    susc_qoa = Compartments(name='susceptible_quarantine')
    compartments.append(susc_qoa)
    expo_qoa = Compartments(name='exposed_quarantine')
    compartments.append(expo_qoa)
    asym_qoa = Compartments(name='asymptomatic_quarantine')
    compartments.append(asym_qoa)
    pre_qoa = Compartments(name='pre-symptomatic_quarantine')
    compartments.append(pre_qoa)
    inf_qoa = Compartments(name='infectious_quarantine')
    compartments.append(inf_qoa)
    rec_qoa = Compartments(name='recovered_quarantine')
    compartments.append(rec_qoa)

    A = Compartments(name='asymptomatic')
    compartments.append(A)
    P = Compartments(name='pre-symptomatic')
    compartments.append(P)
    I = Compartments(name='infectious')
    compartments.append(I)
    Ii = Compartments(name='infectious_i')
    compartments.append(Ii)
    H = Compartments(name='hospitalized')
    compartments.append(H)
    U = Compartments(name='isolated_critical')
    compartments.append(U)

    arg = {o.name: o.value for o in compartments}

    transitions = []
    ct = EconomicReactivation(_compartments=compartments,
                              value_a=rate['GAMMA'],
                              value_b=rate['BETA'],
                              value_c=0.0)
    resp = ct.result(days=DAYS, r0=2.3)
    print(resp)
