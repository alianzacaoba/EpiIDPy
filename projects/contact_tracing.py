import datetime
import time
import numpy as np
from typing import List
from numpy import double
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
        self.contact_matrices = Utils.contact_matrices(file='contact_matrix')
        self._num_comp = len(_compartments)

    def equations(self, x, t, **kwargs):
        try:
            # OA entra como paramento como un lista de ocupación por grupo etario
            dx = np.zeros(self._num_comp, dtype=double)
            contact_matrices = kwargs.get('contact_matrices') if type(kwargs.get('contact_matrices')) is \
                                                                 np.array(dtype=float) else np.array(0, dtype=float)
            age_groups = kwargs.get('age_groups') if type(kwargs.get('age_groups')) is dict() else dict()

            beta = kwargs.get('beta') if type(kwargs.get('beta')) is float else 1.0
            epsilon = kwargs.get('epsilon') if type(kwargs.get('epsilon')) is float else 1.0
            fi = kwargs.get('fi') if type(kwargs.get('fi')) is float else 1.0
            mi = kwargs.get('mi') if type(kwargs.get('mi')) is float else 1.0
            hh = kwargs.get('h') if type(kwargs.get('h')) is float else 1.0
            tt = kwargs.get('tt') if type(kwargs.get('tt')) is float else 1.0
            delta = kwargs.get('delta') if type(kwargs.get('delta')) is float else 1.0
            ma = kwargs.get('ma') if type(kwargs.get('ma')) is float else 1.0
            alfa = kwargs.get('alfa') if type(kwargs.get('alfa')) is float else 1.0
            ceta = kwargs.get('ceta') if type(kwargs.get('ceta')) is float else 1.0
            gg = kwargs.get('g') if type(kwargs.get('g')) is float else 1.0
            uu = kwargs.get('u') if type(kwargs.get('u')) is float else 1.0
            pi = kwargs.get('pi') if type(kwargs.get('pi')) is float else 1.0
            sigma = kwargs.get('sigma') if type(kwargs.get('sigma')) is float else 1.0
            zz = kwargs.get('z') if type(kwargs.get('z')) is float else 1.0
            tao = kwargs.get('tao') if type(kwargs.get('tao')) is float else 1.0
            omega = kwargs.get('omega') if type(kwargs.get('omega')) is float else 1.0

            s_toa, e_toa, p_toa, i_toa, a_toa, r_toa = x
            s_oa, e_oa, p_oa, ii_oa, i_oa, a_oa, r_oa, c_oa, h_oa, u_oa = x

            i = 1.0  # es un compartimento o un valor?
            i_i = 1.0  # es un compartimento o un valor?
            t_oa = s_toa + e_toa + a_toa + p_toa + r_toa
            n = t_oa + s_oa + e_oa + a_oa + p_oa + r_oa + ii_oa + i_oa + c_oa + h_oa

            f = (beta * contact_matrices * (i_oa + a_oa + p_oa)) / n
            f_g = f * (epsilon * s_toa) + s_oa
            f_toa = f_g * tt * zz * (ii_oa + i_oa)

            ds_toa = {1: fi * s_toa, 2: -epsilon * f * s_toa, 3: -mi * s_toa, 4: f_toa * s_toa, 5: -hh * s_toa}
            ds_oa = {1: fi * s_oa, 2: -f * s_oa, 3: -mi * s_oa, 4: hh * s_oa}

            de_toa = {1: f * s_toa, 2: -alfa * e_oa, 3: -mi * e_toa, 4: f_toa * e_toa, 5: -hh * e_toa}
            de_oa = {1: f * s_oa, 2: -alfa * s_oa, 3: -mi * e_oa, 4: -h_oa * e_oa, 5: -hh * e_toa}

            da_toa = {1: alfa * (1 - gg) * e_oa, 2: -ceta * a_toa, 3: -mi * a_toa,
                      4: ma * a_toa, 5: f_toa * a_toa, 6: -hh * a_toa}
            da_oa = {1: alfa * (1 - gg) * e_oa, 2: -ceta * a_oa, 3: -mi * a_oa, 4: ma * a_oa, 5: hh * a_oa}

            dp_toa = {1: alfa * gg * e_oa, 2: -delta * p_toa, 3: -mi * p_toa,
                      4: ma * p_toa, 5: f_toa * p_toa, 6: -hh * p_toa}
            dp_oa = {1: alfa * gg * e_oa, 2: -delta * p_oa, 3: -mi * p_oa,
                     4: ma * p_oa, 5: -hh * p_toa}

            di_ioa = {1: alfa * p_oa, 2: alfa * p_toa, 3: -pi * i_i, 4: f_g * tt * zz * ii_oa}
            di_oa = {1: alfa * p_oa, 2: -pi * i, 3: f_g * tt * zz * i_oa}

            dc_oa = {1: 2 * pi * (1 - (uu + hh)) * (i_i + i), 2: sigma * c_oa}
            dh_oa = {1: 2 * pi * hh * (i_i + i), 2: -tao * h_oa}

            du_oa = {1: 2 * pi * uu * (i_i + i), 2: -omega * u_oa}
            dd_oa = {1: sigma * mi * c_oa, 2: tao * mi * hh, 3: omega * mi * uu}

            dr_toa = {1: ceta * a_oa, 2: f_toa * r_toa, 3: -hh * r_toa}
            dr_oa = {1: sigma * (1 - mi) }

            dx[0] = sum([vs for ks, vs in ds_toa.items()])
            dx[1] = sum([vs for ks, vs in ds_oa.items()])
            dx[2] = sum([vs for ks, vs in de_toa.items()])
            dx[3] = sum([ve for ke, ve in de_oa.items()])
            dx[4] = sum([ve for ke, ve in da_toa.items()])
            dx[5] = sum([ve for ke, ve in da_oa.items()])
            dx[6] = sum([vp for kp, vp in dp_toa.items()])
            dx[7] = sum([vp for kp, vp in dp_oa.items()])
            dx[8] = sum([vi for ki, vi in di_ioa.items()])
            dx[9] = sum([vi for ki, vi in di_oa.items()])
            dx[10] = sum([vc for kc, vc in dc_oa.items()])
            dx[11] = sum([vh for kh, vh in dh_oa.items()])
            dx[12] = sum([vu for ku, vu in du_oa.items()])
            dx[13] = sum([vd for kd, vd in dd_oa.items()])
            dx[14] = sum([vr for kr, vr in dr_toa.items()])
            dx[15] = sum([vr for kr, vr in dr_oa.items()])

            return dx
        except Exception as e:
            print('Error equations: {0}'.format(e))
            return None


if __name__ == "__main__":
    start_processing_s = time.process_time()
    start_time = datetime.datetime.now()
    compartments = []
    s_toa = Compartments(name="Susceptible traced", value=0.0)
    compartments.append(s_toa)
    e_toa = Compartments(name="Exposed traced", value=0.0)
    compartments.append(e_toa)
    a_toa = Compartments(name="Asymptomatic traced", value=0.0)
    compartments.append(a_toa)
    p_toa = Compartments(name="Pre-symptomatic traced", value=0.0)
    compartments.append(p_toa)
    r_toa = Compartments(name="recovered traced", value=0.0)
    compartments.append(r_toa)

    s_oa = Compartments(name="Susceptible", value=0.0)
    compartments.append(s_oa)
    e_oa = Compartments(name="Exposed", value=0.0)
    compartments.append(e_oa)
    a_oa = Compartments(name="Asymptomatic", value=0.0)
    compartments.append(a_oa)
    p_oa = Compartments(name="Pre-symptomatic ", value=0.0)
    compartments.append(p_oa)
    r_oa = Compartments(name="recovered", value=0.0)
    compartments.append(r_oa)
    ii_oa = Compartments(name="Infectious isolate", value=0.0)
    compartments.append(ii_oa)
    i_oa = Compartments(name="Infectious", value=0.0)
    compartments.append(i_oa)
    c_oa = Compartments(name="Inhomecare-Isolated ", value=0.0)
    compartments.append(c_oa)
    h_oa = Compartments(name="Isolated Hospitalization", value=0.0)
    compartments.append(h_oa)
    u_oa = Compartments(name="Isolated-CriticalCare", value=0.0)
    compartments.append(u_oa)

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

    end_processing_s = time.process_time()
    end_processing_ns = time.process_time_ns
    end_time = datetime.datetime.now()
    print('Performance: {0}'.format(end_processing_s - start_processing_s))
    time_diff = (end_time - start_time)
    execution_time = time_diff.total_seconds() * 1000
    print('Execution Time: {0} milliseconds'.format(execution_time))