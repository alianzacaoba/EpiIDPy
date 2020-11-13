from tqdm import tqdm
import datetime
import time
from logic.compartments import Compartments
from logic.utils import Utils
from projects.vaccine.vaccination import Vaccination
from projects.vaccine.calibration import Calibration


class ModelVaccine(object):

    def __init__(self):
        print('Loading input parameters....')

    def run(self, **kwargs):
        initial_population = kwargs.get('initial_population') if type(
            kwargs.get('initial_population')) is dict else dict()
        total_population = kwargs.get('total_population') if type(kwargs.get('total_population')) is dict else dict()
        t_e = kwargs.get('t_e') if type(kwargs.get('t_e')) is dict else dict()
        t_a = kwargs.get('t_a') if type(kwargs.get('t_a')) is dict else dict()
        t_p = kwargs.get('t_p') if type(kwargs.get('t_p')) is dict else dict()
        t_sy = kwargs.get('t_sy') if type(kwargs.get('t_sy')) is dict else dict()
        t_r = kwargs.get('t_r') if type(kwargs.get('t_r')) is dict else dict()
        t_d = kwargs.get('t_d') if type(kwargs.get('t_d')) is dict else dict()
        p_s = kwargs.get('p_s') if type(kwargs.get('p_s')) is dict else dict()
        p_c = kwargs.get('p_c') if type(kwargs.get('p_c')) is dict else dict()
        p_h = kwargs.get('p_h') if type(kwargs.get('p_h')) is dict else dict()
        p_i = kwargs.get('p_i') if type(kwargs.get('p_i')) is dict else dict()
        p_dc = kwargs.get('p_dc') if type(kwargs.get('p_dc')) is dict else dict()
        p_dh = kwargs.get('p_dh') if type(kwargs.get('p_dh')) is dict else dict()
        p_di = kwargs.get('p_di') if type(kwargs.get('p_di')) is dict else dict()
        arrival_rate = kwargs.get('arrival_rate') if type(kwargs.get('arrival_rate')) is dict else dict()
        vaccine_capacities = kwargs.get('vaccine_capacities') if type(
            kwargs.get('vaccine_capacities')) is dict else dict()
        priority_vaccine = kwargs.get('priority_vaccine') if type(
            kwargs.get('priority_vaccine')) is list else list
        try:
            start_processing_s = time.process_time()
            start_time = datetime.datetime.now()
            compartments = []
            su = Compartments(name='Susceptible', value=0)
            compartments.append(su)
            f_1 = Compartments(name='Failure_1', value=0)
            compartments.append(f_1)
            f_2 = Compartments(name='Failure_2', value=0)
            compartments.append(f_2)
            e = Compartments(name='Exposed', value=0)
            compartments.append(e)
            e_f = Compartments(name='Exposed_Failure', value=0)
            compartments.append(e_f)
            a = Compartments(name='Asymptomatic', value=0)
            compartments.append(a)
            a_f = Compartments(name='Asymptomatic_Failure', value=0)
            compartments.append(a_f)
            p = Compartments(name='Pre-symptomatic', value=0)
            compartments.append(p)
            sy = Compartments(name='Symptomatic ', value=0)
            compartments.append(sy)
            c = Compartments(name='Home ', value=0)
            compartments.append(c)
            h = Compartments(name='Hospitalization', value=0)
            compartments.append(h)
            i = Compartments(name='ICU', value=0)
            compartments.append(i)
            r = Compartments(name='Recovered', value=0)
            compartments.append(r)
            r_a = Compartments(name='Recovered_Asymptomatic', value=0)
            compartments.append(r)
            v_1 = Compartments(name='Vaccination_1', value=0)
            compartments.append(v_1)
            v_2 = Compartments(name='Vaccination_2', value=0)
            compartments.append(v_2)
            d = Compartments(name='Dead', value=0)
            compartments.append(d)
            cases = Compartments(name='Cases', value=0)
            compartments.append(cases)

            contact_matrix = Utils.contact_matrices(file='contact_matrix', delimiter=',')
            print('Calculated Vaccination.....')
            result_vd = {}
            result_for_cal = list()
            sim_length = 230
            for i in range(sim_length):
                result_for_cal.append(0)
            setting = {'contact_matrix': contact_matrix}
            for dept, age_work_health in tqdm(initial_population.items()):
                total_population_dept = total_population[dept]
                age_vd = dict()
                for ka, va in dict(age_work_health).items():
                    work_vd = dict()
                    for kw, vw in dict(va).items():
                        health_vd = dict()
                        for kh, vh in dict(vw).items():
                            su.value = vh * (1 - sus_initial['ALL'])
                            r_a.value = vh * (1 - sus_initial['ALL'])
                            vaccine = Vaccination(_compartments=compartments, r0=0.0)
                            setting.update({'calibration': True, 'arrival_rate': arrival_rate[dept], 'beta': 0.5,
                                            'age_group': ka, 'work_group': kw, 'health_group': kh,
                                            'epsilon_1': 0.0, 'epsilon_2': 0.0, 't_e': t_e['ALL'],
                                            't_a': t_a['ALL'], 't_p': t_p['ALL'], 't_sy': t_sy['ALL'],
                                            't_r': t_r['ALL'], 't_d': t_d[ka], 'p_s': p_s[ka], 'p_c': p_c[ka],
                                            'p_h': p_h[ka], 'p_i': (1 - (p_c[ka] + p_h[ka])),
                                            'p_dc': p_dc[ka][kh], 'p_dh': p_dh[ka][kh], 'p_di': p_di[ka][kh],
                                            'total_population': total_population_dept,
                                            'initial_population': age_work_health,
                                            'priority_vaccine': priority_vaccine,
                                            'vaccine_capacities': vaccine_capacities[dept]})
                            resp = vaccine.run(days=sim_length, **setting)
                            for t in range(sim_length):
                                result_for_cal[t] += resp['Cases'][t]
                            health_vd[kh] = resp
                        work_vd[kw] = health_vd
                    age_vd[ka] = work_vd
                result_vd[dept] = age_vd


            end_processing_s = time.process_time()
            end_processing_ns = time.process_time_ns
            end_time = datetime.datetime.now()
            print('Performance: {0}'.format(end_processing_s - start_processing_s))
            time_diff = (end_time - start_time)
            execution_time = time_diff.total_seconds()
            mm = int(execution_time / 60)
            ss = int(execution_time % 60)
            print('Execution Time: {0} minutes {1} seconds'.format(mm, ss))
            print('Execution Time: {0} milliseconds'.format(execution_time * 1000))
            Utils.save('vaccination', result_vd)

        except Exception as e:
            print('Error vaccine_assignment: {0}'.format(e))
            return None


if __name__ == "__main__":
    ut = Utils()
    initial_population = ut.initial_population(file='initial_population', delimiter=';')
    total_population = ut.total_population(file='initial_population', delimiter=';')
    vaccine_capacities = ut.region_capacities(file='region_capacities')
    priority_vaccine = ut.priority_vaccine(file='priority_vaccines', delimiter=';', scenario=1)
    arrival_rate = ut.arrival_rate(file='arrival_rate', delimiter=';', filter='CALCULATED_RATE')

    sus_initial = ut.probabilities(file='input_probabilities', delimiter=';',
                                   parameter_1='InitialSus', parameter_2='ALL', filter='BASE_VALUE')
    p_s = ut.probabilities(file='input_probabilities', delimiter=';',
                           parameter_1='ExposedToPresymp', parameter_2='ALL', filter='BASE_VALUE')
    p_c = ut.probabilities(file='input_probabilities', delimiter=';',
                           parameter_1='SympToHouse', parameter_2='ALL', filter='BASE_VALUE')
    p_h = ut.probabilities(file='input_probabilities', delimiter=';',
                           parameter_1='SympToHospital', parameter_2='ALL', filter='BASE_VALUE')
    p_dc = ut.probabilities(file='input_probabilities', delimiter=';',
                            parameter_1='HouseToDeath', filter='BASE_VALUE')
    p_dh = ut.probabilities(file='input_probabilities', delimiter=';',
                            parameter_1='HospitalToDeath', filter='BASE_VALUE')
    p_di = ut.probabilities(file='input_probabilities', delimiter=';',
                            parameter_1='ICUToDeath', filter='BASE_VALUE')

    t_e = ut.input_time(file='input_time', delimiter=';', parameter='IncubationTime', filter='BASE_VALUE')
    t_a = ut.input_time(file='input_time', delimiter=';', parameter='RecoveryTimeAsymptomatic', filter='BASE_VALUE')
    t_p = ut.input_time(file='input_time', delimiter=';', parameter='PresymptomaticTime', filter='BASE_VALUE')
    t_sy = ut.input_time(file='input_time', delimiter=';', parameter='SymptomsToTreatment', filter='BASE_VALUE')
    t_r = ut.input_time(file='input_time', delimiter=';', parameter='TreatmentToRecovery', filter='BASE_VALUE')
    t_d = ut.input_time(file='input_time', delimiter=';', parameter='TimeToDeath', filter='BASE_VALUE')

    kwargs = {'initial_population': initial_population, 'total_population': total_population,
              'vaccine_capacities': vaccine_capacities, 'priority_vaccine': priority_vaccine,
              'arrival_rate': arrival_rate, 'sus_initial': sus_initial, 'p_s': p_s, 'p_c': p_c,
              'p_h': p_h, 'p_dc': p_dc, 'p_dh': p_dh, 'p_di': p_di, 't_e': t_e, 't_a': t_a,
              't_p': t_p, 't_sy': t_sy, 't_r': t_r, 't_d': t_d}
    mv = ModelVaccine()
    mv.run(**kwargs)
