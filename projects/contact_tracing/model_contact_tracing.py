import datetime
import time
import numpy as np
from typing import List
from numpy import double
from tqdm import tqdm
from logic.compartments import Compartments
from logic.disease_model import DiseaseModel
from logic.utils import Utils
from data.input.input_time import input_time


class ModelContactTracing(DiseaseModel):
    """ Class used to represent an Contact Tracing disease model """

    def __init__(self, _compartments: List[Compartments], r0: float):
        """
        Initialize the run of contact tracing disease model
        """
        super().__init__(_compartments, r0=r0)
        self._num_comp = len(_compartments)

    def equations(self, x, t, **kwargs):
        """Time equations of the state vector.
        :param x: The compartment vector (array_like)
        :type x: Object Compartments
        :param t: time (scalar)
        :type t: int
        Keyword kwargs:
            :contact_matrix dict: contact matrix by age group.
            :population dict: dictionary by departament and age group.
            :total_population float: total population by departament.
        :returns: Disease model equations.
        :rtype: dict
        """
        try:
            # OA entra como paramento como un lista de ocupación por grupo etario
            dx = np.zeros(self._num_comp, dtype=double)
            contact_matrix = kwargs.get('contact_matrix') if type(kwargs.get('contact_matrix')) is dict else dict()
            population = kwargs.get('population') if type(kwargs.get('population')) is dict else dict()
            total_population = kwargs.get('total_population') if type(kwargs.get('total_population')) is float else 1.0
            age = kwargs.get('age') if type(kwargs.get('age')) is str else str

            mi_ae = kwargs.get('mi_ae') if type(kwargs.get('mi_ae')) is float else 1.0
            mi_ac = kwargs.get('mi_ac') if type(kwargs.get('mi_ac')) is float else 1.0
            mi_ah = kwargs.get('mi_ah') if type(kwargs.get('mi_ah')) is float else 1.0
            mi_au = kwargs.get('mi_au') if type(kwargs.get('mi_au')) is float else 1.0

            beta = kwargs.get('beta') if type(kwargs.get('beta')) is float else 0.0
            fi = kwargs.get('fi') if type(kwargs.get('fi')) is float else 0.0
            alfa = kwargs.get('alfa') if type(kwargs.get('alfa')) is float else 0.0

            ro = kwargs.get('ro') if type(kwargs.get('ro')) is float else 0.0
            tt = kwargs.get('tt') if type(kwargs.get('tt')) is float else 0.0
            delta = kwargs.get('delta') if type(kwargs.get('delta')) is float else 0.0

            ceta = kwargs.get('ceta') if type(kwargs.get('ceta')) is float else 0.0
            gg = kwargs.get('g') if type(kwargs.get('g')) is float else 0.0
            hh = kwargs.get('hh') if type(kwargs.get('hh')) is float else 0.0
            uu = kwargs.get('uu') if type(kwargs.get('uu')) is float else 0.0
            pi = kwargs.get('pi') if type(kwargs.get('pi')) is float else 0.0
            sigma = kwargs.get('sigma') if type(kwargs.get('sigma')) is float else 0.0
            tao = kwargs.get('tao') if type(kwargs.get('tao')) is float else 0.0
            omega = kwargs.get('omega') if type(kwargs.get('omega')) is float else 0.0

            s_oa, s_toa, e_oa, e_toa, a_oa, a_toa, p_oa, p_toa, ii_oa, i_oa, c_oa, h_oa, u_oa, d_oa, r_oa, r_toa = x

            t_oa = s_toa + e_toa + a_toa + p_toa + r_toa
            population_list = [float(v) for _, v in population.items()]

            prod_matrix = [a * b for a, b in zip(contact_matrix[age], population_list)]
            prod_matrix = sum(prod_matrix)

            f = (beta * prod_matrix * (i_oa + a_oa + p_oa)) / total_population
            f_oa = f * (s_toa + s_oa)
            f_toa = f_oa * tt

            ds_toa = {1: fi * s_toa, 2: -f * s_toa, 3: -mi_ae * s_toa, 4: f_toa * s_toa, 5: -ro * s_toa}
            ds_oa = {1: fi * s_oa, 2: -f * s_oa, 3: -mi_ae * s_oa, 4: ro * s_oa}

            de_toa = {1: f * s_toa, 2: -alfa * e_oa, 3: -mi_ae * e_toa, 4: f_toa * e_toa, 5: -ro * e_toa}
            de_oa = {1: f * s_oa, 2: -alfa * e_oa, 3: -mi_ae * e_oa, 4: ro * e_toa}

            da_toa = {1: alfa * (1 - gg) * e_oa, 2: -ceta * a_toa, 3: -mi_ae * a_toa, 4: f_toa * a_toa, 5: -ro * a_toa}
            da_oa = {1: alfa * (1 - gg) * e_oa, 2: -ceta * a_oa, 3: -mi_ae * a_oa, 4: ro * a_oa}

            dp_toa = {1: alfa * gg * e_oa, 2: -delta * p_toa, 3: -mi_ae * p_toa,
                      4: f_toa * p_toa, 6: -ro * p_toa}
            dp_oa = {1: alfa * gg * e_oa, 2: -delta * p_oa, 3: -mi_ae * p_oa, 4: -ro * p_toa}

            di_ioa = {1: alfa * p_oa, 2: alfa * p_toa, 3: -pi * ii_oa, 4: -f_oa * tt * ii_oa}
            di_oa = {1: alfa * p_oa, 2: -pi * i_oa, 3: -f_oa * tt * i_oa}

            dc_oa = {1: 2 * pi * (1 - (uu + hh)) * (ii_oa + i_oa), 2: -sigma * c_oa}
            dh_oa = {1: 2 * pi * ro * hh * (ii_oa + i_oa), 2: -tao * h_oa}

            du_oa = {1: 2 * pi * uu * (ii_oa + i_oa), 2: -omega * u_oa}
            dd_oa = {1: sigma * mi_ac * c_oa, 2: tao * mi_ah * h_oa, 3: omega * mi_au * u_oa}

            dr_toa = {1: ceta * a_oa, 2: f_toa * r_toa, 3: -ro * r_toa}

            dr_oa = {1: sigma * (1 - mi_ac) * c_oa, 2: tao * (1 - mi_ah) * h_oa,
                     3: omega * (1 - mi_au) * u_oa, 4: ro * r_toa}

            dx[0] = sum([vs for ks, vs in ds_oa.items()])
            dx[1] = sum([vs for ks, vs in ds_toa.items()])
            dx[2] = sum([ve for ke, ve in de_oa.items()])
            dx[3] = sum([vs for ks, vs in de_toa.items()])
            dx[4] = sum([ve for ke, ve in da_oa.items()])
            dx[5] = sum([ve for ke, ve in da_toa.items()])
            dx[6] = sum([vp for kp, vp in dp_oa.items()])
            dx[7] = sum([vp for kp, vp in dp_toa.items()])
            dx[8] = sum([vi for ki, vi in di_ioa.items()])
            dx[9] = sum([vi for ki, vi in di_oa.items()])
            dx[10] = sum([vc for kc, vc in dc_oa.items()])
            dx[11] = sum([vh for kh, vh in dh_oa.items()])
            dx[12] = sum([vu for ku, vu in du_oa.items()])
            dx[13] = sum([vd for kd, vd in dd_oa.items()])
            dx[14] = sum([vr for kr, vr in dr_oa.items()])
            dx[15] = sum([vr for kr, vr in dr_toa.items()])

            return dx
        except Exception as e:
            print('Error equations: {0}'.format(e))
            return None


if __name__ == "__main__":
    start_processing_s = time.process_time()
    start_time = datetime.datetime.now()
    ut = Utils()
    initial_population = ut.initial_population_ct(file='initial_population', delimiter=';')
    total_population = ut.total_population(file='initial_population', delimiter=';')
    mi_ae = ut.probabilities(parameter_1='MortalityAllCauseMortality', parameter_2='ALL')
    mi_ac = ut.probabilities(parameter_1='MortalityHomecare', parameter_2='ALL')
    mi_ah = ut.probabilities(parameter_1='MortalityHospitalized', parameter_2='ALL')
    mi_au = ut.probabilities(parameter_1='MortalityCriticalCare', parameter_2='ALL')
    ro = ut.probabilities(parameter_1='TraceNoTrace', parameter_2='ALL')
    death_hospitalized = ut.probabilities(parameter_1='SymptomaticHospitalized', parameter_2='ALL')
    death_critical_care = ut.probabilities(parameter_1='SymptomaticCriticalCarePopulation', parameter_2='ALL')

    contact_matrix = Utils.contact_matrices(file='contact_matrix', delimiter=',')

    compartments = []
    s_oa = Compartments(name="Susceptible", value=0.0)
    compartments.append(s_oa)
    s_toa = Compartments(name="Susceptible traced", value=0.0)
    compartments.append(s_toa)
    e_oa = Compartments(name="Exposed", value=0.0)
    compartments.append(e_oa)
    e_toa = Compartments(name="Exposed traced", value=0.0)
    compartments.append(e_toa)
    a_oa = Compartments(name="Asymptomatic", value=0.0)
    compartments.append(a_oa)
    a_toa = Compartments(name="Asymptomatic traced", value=0.0)
    compartments.append(a_toa)
    p_oa = Compartments(name="Pre-symptomatic", value=0.0)
    compartments.append(p_oa)
    p_toa = Compartments(name="Pre-symptomatic traced", value=0.0)
    compartments.append(p_toa)
    ii_oa = Compartments(name="Infectious isolate", value=0.0)
    compartments.append(ii_oa)
    i_oa = Compartments(name="Infectious", value=0.0)
    compartments.append(i_oa)
    c_oa = Compartments(name="Inhomecare-Isolated", value=0.0)
    compartments.append(c_oa)
    h_oa = Compartments(name="Isolated Hospitalization", value=0.0)
    compartments.append(h_oa)
    u_oa = Compartments(name="Isolated-CriticalCare", value=0.0)
    compartments.append(u_oa)
    d_oa = Compartments(name="Dead", value=0.0)
    compartments.append(d_oa)
    r_oa = Compartments(name="Recovered", value=0.0)
    compartments.append(r_oa)
    r_toa = Compartments(name="recovered traced", value=0.0)
    compartments.append(r_toa)

    kwargs = {'beta': input_time['PerCapitaTransmissionRate'],
              'fi': (0.012/365),
              'alfa': input_time['ExposedToAsymptomatic'],
              'g': input_time['PresymptomaticTime'],
              'ceta': input_time['AsymptomaticToRecovered'],
              'delta': input_time['PresymptomaticToInfectious'],
              'pi': input_time['InfectiousToHomecareCare'],
              'sigma': input_time['HomecareToDeath'],
              'tt': 10.0,
              'ro': ro,
              'tao': input_time['HospitalizedToDeath'],
              'omega': input_time['CriticalCareToDeath'],
              'contact_matrix': contact_matrix}

    result = dict()
    for dept, work_group in tqdm(initial_population.items()):
        total_population_dept = sum(dict(total_population[dept]).values())
        work_group_ct = dict()
        ct = ModelContactTracing(_compartments=compartments, r0=0.0)
        work_ct = dict()
        for kw, vv in dict(work_group).items():
            age_ct = dict()
            for ka, va in dict(vv).items():
                s_oa.value = va
                kwargs.update({'age': ka,
                               'population': vv,
                               'hh': death_hospitalized[ka],
                               'uu': death_critical_care[ka],
                               'mi_ae': mi_ae[ka], 'mi_ac': mi_ac[ka],
                               'mi_ah': mi_ah[ka], 'mi_au': mi_au[ka],
                               'total_population': total_population_dept})
                resp = ct.run(days=100, **kwargs)
                age_ct[ka] = resp
            work_ct[kw] = age_ct
        result[dept] = work_ct
    end_processing_s = time.process_time()
    end_processing_ns = time.process_time_ns
    end_time = datetime.datetime.now()
    print('Performance: {0}'.format(end_processing_s - start_processing_s))
    time_diff = (end_time - start_time)
    execution_time = time_diff.total_seconds() * 1000
    print('Execution Time: {0} milliseconds'.format(execution_time))
    Utils.save('contact_tracing', result)
    Utils.export_excel_ct('contact_tracing', result)