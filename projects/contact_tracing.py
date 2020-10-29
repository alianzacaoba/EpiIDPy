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
        self.contact_matrices = Utils.contact_matrices(file='total_contact_matrix')
        self._num_comp = len(_compartments)

    def equations(self, x, t, **kwargs):
        try:
            # OA entra como paramento como un lista de ocupación por grupo etario
            dx = np.zeros(self._num_comp, dtype=double)
            contact_matrices = kwargs.get('contact_matrices') if type(kwargs.get('contact_matrices')) is \
                                                                 np.array(dtype=float) else np.array(0, dtype=float)
            age_groups = kwargs.get('age_groups') if type(kwargs.get('age_groups')) is dict() else dict()

            beta = kwargs.get('beta') if type(kwargs.get('beta')) is float else 1.0
            delta = kwargs.get('delta') if type(kwargs.get('delta')) is float else 1.0
            epsilon = kwargs.get('epsilon') if type(kwargs.get('epsilon')) is float else 1.0
            mi = kwargs.get('mi') if type(kwargs.get('mi')) is float else 1.0
            ma = kwargs.get('ma') if type(kwargs.get('ma')) is float else 1.0
            fi = kwargs.get('fi') if type(kwargs.get('fi')) is float else 1.0
            alfa = kwargs.get('alfa') if type(kwargs.get('alfa')) is float else 1.0
            ceta = kwargs.get('ceta') if type(kwargs.get('ceta')) is float else 1.0
            gg = kwargs.get('g') if type(kwargs.get('g')) is float else 1.0
            pa = kwargs.get('pa') if type(kwargs.get('pa')) is float else 1.0
            uu = kwargs.get('u') if type(kwargs.get('u')) is float else 1.0
            hh = kwargs.get('h') if type(kwargs.get('h')) is float else 1.0
            pi = kwargs.get('pi') if type(kwargs.get('pi')) is float else 1.0
            sigma = kwargs.get('sigma') if type(kwargs.get('sigma')) is float else 1.0
            zz = kwargs.get('z') if type(kwargs.get('z')) is float else 1.0
            tao = kwargs.get('tao') if type(kwargs.get('tao')) is float else 1.0
            omega = kwargs.get('omega') if type(kwargs.get('omega')) is float else 1.0

            s_q, e_q, p_q, i_q, a_q, r_q = x
            s_qt, e_qt, p_qt, i_qt, a_qt, r_qt = x
            s, e, p, ii, i, a, r, c, h, u = x

            q = s_q + e_q + a_q + p_q + r_q
            tt = s_qt + e_qt + a_qt + p_qt + r_qt
            n = q + tt + s + e + a + p + r + ii + i + c + h + u

            f = contact_matrices * (beta * (i + a + p)) / n
            F = f * ((epsilon * (s_qt + s_q)) + s)
            f_t = F * tt * (ii + i)

            ds_qt = {1: fi * s_qt, 2: -epsilon * f * s_qt, 3: -mi * s_qt, 4: ma * s_qt, 5: f_t * s_qt}
            ds_q = {1: fi * s_q, 2: -epsilon * f * s_q, 3: -mi * s_q, 4: ma * s_q, 5: h * s, 6: f_t * s_q}
            ds = {1: fi * s, 2: -f * s, 3: -mi * s, 4: ma * s, 5: h * s}

            de_qt = {1: f * s_qt, 2: f * s_q, 3: -alfa * e_q, 4: -mi * e_q, 5: ma * e_q, 6: f_t * e_qt}
            de_q = {1: f * s_qt, 2: f * s_q, 3: -alfa * e_q, 4: -mi * e_q, 5: ma * e_q, 6: h * e, 7: f_t * e_q}
            de = {1: f * s, 2: -alfa * s, 3: -mi * e, 4: ma * e, 5: -h * e, 6: -ma * e_q, 7: -h * e}

            da = {1: alfa * (1 - gg) * e, 2: -ceta * a, 3: -mi * a, 4: ma * a, 5: -h * a}
            da_qt = {1: alfa * (1 - gg) * e_q, 2: -ceta * a_q, 3: -mi * a_q, 4: ma * a_q, 5: f_t * a_qt}
            da_q = {1: alfa * (1 - gg) * e_q, 2: -ceta * a_q, 3: -mi * a_q, 4: ma * a_q, 5: -h * a, 6: f_t * a_q}

            dp_qt = {1: alfa * gg * e_q, 2: -delta * p_q, 3: -mi * p_q, 4: ma * p_q, 5: f_t * p_qt}
            dp_q = {1: alfa * gg * e_q, 2: -delta * p_q, 3: -mi * p_q, 4: ma * p_q, 5: h * p, 6: f_t * p_q}
            dp = {1: alfa * gg * e, 2: -delta * p, 3: -mi * p, 4: ma * p, 5: -h * p}

            di_i = {1: delta * pa * p, 2: delta * p_q, 3: -pi * ii, 4: ma * i, 5: F * tt * zz * i}
            di = {1: delta * (1 - pa) * p, 2: -pi * i, 3: ma * i, 4: F * tt * ii}

            dc = {1: 2 * pi * (1 - (uu + hh)) * (ii + i), 2: sigma * c}
            dh = {1: 2 * pi * h * (ii + i), 2: tao * h}
            du = {1: 2 * pi * h * (ii + i), 2: omega * h}
            dd = {1: sigma * mi * c, 2: tao * mi * h, 3: omega * mi * u}
            dr_q = {1: ceta * a, 2: ma * r_q, 3: F * r_q}
            dr_qt = {1: ceta * a, 2: ma * r_q, 3: h * r, 4: F * r_q}
            dr = {1: sigma * (1 - mi) * c, 2: tao * (1 - mi) * h, 3: omega * (1 - mi) * u, 4: ma * r, 5: h * r }

            dx[0] = sum([vs for ks, vs in ds_qt.items()])
            dx[1] = sum([vs for ks, vs in ds_q.items()])
            dx[2] = sum([vs for ks, vs in ds.items()])
            dx[3] = sum([ve for ke, ve in de_qt.items()])
            dx[4] = sum([ve for ke, ve in de_q.items()])
            dx[5] = sum([ve for ke, ve in de.items()])
            dx[6] = sum([va for ka, va in da.items()])
            dx[7] = sum([va for ka, va in da_qt.items()])
            dx[8] = sum([va for ka, va in da_q.items()])
            dx[9] = sum([vp for kp, vp in dp_qt.items()])
            dx[10] = sum([vp for kp, vp in dp_q.items()])
            dx[11] = sum([vp for kp, vp in dp.items()])
            dx[12] = sum([vi for ki, vi in di_i.items()])
            dx[13] = sum([vi for ki, vi in di.items()])
            dx[14] = sum([vc for kc, vc in dc.items()])
            dx[15] = sum([vh for kh, vh in dh.items()])
            dx[16] = sum([vu for ku, vu in du.items()])
            dx[17] = sum([vd for kd, vd in dd.items()])
            dx[18] = sum([vr for kr, vr in dr_q.items()])
            dx[19] = sum([vr for kr, vr in dr_qt.items()])
            dx[20] = sum([vr for kr, vr in dr.items()])
            return dx
        except Exception as e:
            print('Error equations: {0}'.format(e))
            return None


if __name__ == "__main__":
    start = timer()
    compartments = []
    s_q = Compartments(name="Susceptible in quarantine", value=0.0)
    compartments.append(s_q)
    e_q = Compartments(name="Exposed in quarantine", value=0.0)
    compartments.append(e_q)
    p_q = Compartments(name="Pre-symptomatic in quarantine", value=0.0)
    compartments.append(p_q)
    i_q = Compartments(name="Infectious in quarantine", value=0.0)
    compartments.append(i_q)
    a_q = Compartments(name="Asymptomatic in quarantine", value=0.0)
    compartments.append(a_q)
    r_q = Compartments(name="recovered in quarantine", value=0.0)
    compartments.append(r_q)

    s_qt = Compartments(name="Susceptible in quarantine traced", value=0.0)
    compartments.append(s_qt)
    e_qt = Compartments(name="Exposed in quarantine traced", value=0.0)
    compartments.append(e_qt)
    p_qt = Compartments(name="Pre-symptomatic in quarantine traced", value=0.0)
    compartments.append(p_qt)
    i_qt = Compartments(name="Infectious in quarantine traced", value=0.0)
    compartments.append(i_qt)
    a_qt = Compartments(name="Asymptomatic in quarantine traced", value=0.0)
    compartments.append(a_qt)
    r_qt = Compartments(name="recovered in quarantine traced", value=0.0)
    compartments.append(r_qt)

    s = Compartments(name="Susceptible", value=0.0)
    compartments.append(s)
    e = Compartments(name="Exposed", value=0.0)
    compartments.append(e)
    p = Compartments(name="Pre-symptomatic ", value=0.0)
    compartments.append(p)
    i = Compartments(name="Infectious", value=0.0)
    compartments.append(i)
    ii = Compartments(name="Infectious isolate", value=0.0)
    compartments.append(ii)
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
              'beta': 0.0, 'ceta': 0.0, 'pa': 0.0, 'g': 0.0, 'u': 0.0,
              'h': 0.0, 'pi': 0.0, 'sigma': 0.0, 't': 0.0, 'z': 0.0,
              'tao': 0.0, 'omega': 0.0, 'contact_matrices': ct.contact_matrices}
    for dept, age_groups in ct.population.items():
        temp = dict()
        for age, value in dict(age_groups).items():
            kwargs.update({'age_groups': value})
            resp = ct.run(days=100, **kwargs)
            temp[age] = resp
        result[dept] = temp
    print(result)
    Utils.save('contact_tracing', result)
    end = timer()
    print('Time processing: {0}'.format(timedelta(seconds=end - start)))
