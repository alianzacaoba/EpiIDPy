from timeit import default_timer as timer
from datetime import timedelta
import numpy as np
from typing import List
from numpy import double
from config.contact_settings import DAYS, GAMMA, BETA
from logic.compartments import Compartments
from logic.disease_model import DiseaseModel
from logic.utils import Utils


class ContactTracing(DiseaseModel):
    """ Class used to represent an Contact Tracing disease model """

    def __init__(self, _compartments: List[Compartments], r0: float, value_b: float = 0.0, value_c: float = 0.0):
        """
        Initialize the run of contact tracing disease model
        """
        super().__init__(_compartments, r0=r0, value_b=value_b, value_c=value_c)
        self.population = Utils.population(file='population', year=2020)
        self._num_comp = len(_compartments)

    def equations(self, x, t, **kwargs):
        try:
            dx = np.zeros(self._num_comp, dtype=double)
            beta = kwargs.get('beta') if type(kwargs.get('beta')) is float else 1.0
            delta = kwargs.get('delta') if type(kwargs.get('delta')) is float else 1.0
            epsilon = kwargs.get('epsilon') if type(kwargs.get('epsilon')) is float else 1.0
            mi = kwargs.get('mi') if type(kwargs.get('mi')) is float else 1.0
            ma = kwargs.get('ma') if type(kwargs.get('ma')) is float else 1.0
            fi = kwargs.get('fi') if type(kwargs.get('fi')) is float else 1.0
            alfa = kwargs.get('alfa') if type(kwargs.get('alfa')) is float else 1.0
            ceta = kwargs.get('ceta') if type(kwargs.get('ceta')) is float else 1.0

            s_qoa, e_qoa, p_qoa, i_qoa, a_qoa, r_qoa = x
            s_qtoa, e_qtoa, p_qtoa, i_qtoa, a_qtoa, r_qtoa = x
            s_oa, e_oa, p_oa, i_oa, a_oa, r_oa, i_oa, c_oa, h_oa, u_oa = x
            q_oa = s_qoa + e_qoa + a_qoa + p_qoa + r_qoa
            t_oa = s_qtoa + e_qtoa + p_qtoa + i_qtoa + a_qtoa + r_qtoa
            n = q_oa + t_oa + s_oa + e_oa + p_oa + i_oa + a_oa + r_oa + i_oa + c_oa + h_oa + u_oa

            f = (beta * (i_oa + a_oa + p_oa)) / n
            f_toa = 1.0
            g = 1.0

            ds_qtoa = {1: fi * s_qtoa, 2: -epsilon * f * s_qtoa,
                       3: -mi * s_qtoa, 4: ma * s_qtoa, 5: f_toa * s_qtoa}
            ds_qoa = {1: fi * s_qoa, 2: -epsilon * f * s_qoa,
                      3: -mi * s_qoa, 4: ma * s_qoa, 5: f_toa * s_qoa}
            ds_oa = {1: fi * s_oa, 2: -f * s_qtoa, 3: -mi * s_oa,
                     4: ma * s_oa, 5: h_oa * s_qtoa}

            de_qtoa = {1: f * s_qtoa, 2: f * s_qoa, 3: -alfa * e_qoa,
                       4: mi * e_qoa, 5: ma * e_qoa, 6: f_toa * e_qtoa}
            de_qoa = {1: f * s_qtoa, 2: f * s_qoa, 3: -alfa * e_qoa,
                      4: -mi * e_qoa, 5: ma * e_qoa, 6: h_oa * e_oa, 7: f_toa * e_qoa}
            de_oa = {1: f * s_oa, 2: -f * s_oa, 3: mi * e_oa,
                     4: ma * e_oa, 5: -h_oa * e_oa, 6: -ma * e_oa}

            da_oa = {1: alfa * (1 - g) * e_oa, 2: -ceta * a_oa, 3: -mi * a_oa,
                     4: ma * a_oa, 5: -h_oa * a_oa}
            da_qtoa = {1: alfa * (1 - g) * e_qoa, 2: -ceta * a_qoa, 3: -mi * a_qoa,
                       4: ma * a_qoa, 5: -h_oa * a_qoa}
            da_qoa = {1: alfa * (1 - g) * e_qoa, 2: -ceta * a_qoa, 3: -mi * a_qoa,
                       4: ma * a_qoa, 5: -h_oa * a_oa, 6: f_toa * a_qoa}

            dp_qtoa = {1: alfa * g * e_qoa, 2: -delta * p_qoa, 3: -mi * p_qoa,
                       4: ma * p_qoa, 5: f_toa * p_qtoa}
            dp_qoa = {1: alfa * g * e_qoa, 2: -delta * p_qoa, 3: -mi * p_qoa,
                       4: ma * p_qoa, 5: h_oa * p_oa, 6: f_toa * p_qtoa}
            dp_oa = {1: alfa * g * e_oa, 2: -delta * p_oa, 3: -mi * p_oa,
                     4: ma * p_oa, 5: -h_oa * p_oa}

            dx[0] = sum([vs for ks, vs in ds_qtoa.items()])
            dx[1] = sum([vs for ks, vs in ds_qoa.items()])
            dx[2] = sum([vs for ks, vs in ds_oa.items()])
            dx[3] = sum([ve for ke, ve in de_qtoa.items()])
            dx[4] = sum([ve for ke, ve in de_qoa.items()])
            dx[5] = sum([ve for ke, ve in de_oa.items()])
            dx[6] = sum([va for ka, va in da_oa.items()])
            dx[7] = sum([va for ka, va in da_qtoa.items()])
            dx[8] = sum([va for ka, va in da_qoa.items()])
            dx[9] = sum([va for ka, va in da_qoa.items()])
            dx[10] = sum([vp for kp, vp in dp_qtoa.items()])
            dx[11] = sum([vp for kp, vp in dp_qoa.items()])
            dx[12] = sum([vp for kp, vp in dp_oa.items()])

            return dx
        except Exception as e:
            print('Error equations: {0}'.format(e))
            return None


if __name__ == "__main__":
    start = timer()
    compartments = []
    s_qoa = Compartments(name="Susceptible in quarantine", value=0.0)
    compartments.append(s_qoa)
    e_qoa = Compartments(name="Exposed in quarantine", value=0.0)
    compartments.append(e_qoa)
    p_qoa = Compartments(name="Pre-symptomatic in quarantine", value=0.0)
    compartments.append(p_qoa)
    i_qoa = Compartments(name="Infectious in quarantine", value=0.0)
    compartments.append(i_qoa)
    a_qoa = Compartments(name="Asymptomatic in quarantine", value=0.0)
    compartments.append(a_qoa)
    r_qoa = Compartments(name="recovered in quarantine", value=0.0)
    compartments.append(r_qoa)

    s_qtoa = Compartments(name="Susceptible in quarantine traced", value=0.0)
    compartments.append(s_qtoa)
    e_qtoa = Compartments(name="Exposed in quarantine traced", value=0.0)
    compartments.append(e_qtoa)
    p_qtoa = Compartments(name="Pre-symptomatic in quarantine traced", value=0.0)
    compartments.append(p_qtoa)
    i_qtoa = Compartments(name="Infectious in quarantine traced", value=0.0)
    compartments.append(i_qtoa)
    a_qtoa = Compartments(name="Asymptomatic in quarantine traced", value=0.0)
    compartments.append(a_qtoa)
    r_qtoa = Compartments(name="recovered in quarantine traced", value=0.0)
    compartments.append(r_qtoa)

    s = Compartments(name="Susceptible", value=0.0)
    compartments.append(s)
    e = Compartments(name="Exposed", value=0.0)
    compartments.append(e)
    p = Compartments(name="Pre-symptomatic ", value=0.0)
    compartments.append(p)
    i = Compartments(name="Infectious", value=0.0)
    compartments.append(i)
    a = Compartments(name="Asymptomatic", value=0.0)
    compartments.append(a)
    c = Compartments(name="Inhomecare-Isolated ", value=0.0)
    compartments.append(c)
    h = Compartments(name="Isolated-hospitalization", value=0.0)
    compartments.append(h)
    u = Compartments(name="Isolated-CriticalCare", value=0.0)
    compartments.append(u)
    r = Compartments(name="recovered", value=0.0)
    compartments.append(r)
    d = Compartments(name="dead", value=0.0)
    compartments.append(d)

    ct = ContactTracing(_compartments=compartments, r0=0.0)
    result = dict()
    kwargs = {'fi': 0.4, 'delta': 5.0, 'epsilon': 0.0, 'mi': 0.0, 'ma': 0.0,
              'beta': 0.0, 'ceta': 0.0}
    for dept, age_groups in ct.population.items():
        temp = dict()
        for age, value in dict(age_groups).items():
            resp = ct.run(days=100)
            temp[age] = resp
        result[dept] = temp
    print(result)
    Utils.save('contact_tracing', result)
    end = timer()
    print('Time processing: {0}'.format(timedelta(seconds=end - start)))
