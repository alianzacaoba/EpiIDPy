from tqdm import tqdm
import datetime
import time
from logic.compartments import Compartments
from logic.utils import Utils
from projects.vaccine.vaccination import Vaccination

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

print('Loading input parameters....')
initial_population = Utils.initial_population(file='initial_population', delimiter=';')
total_population = Utils.total_population(file='initial_population', delimiter=';')
vaccine_capacities = Utils.region_capacities(file='region_capacities')
priority_vaccine = Utils.priority_vaccine(file='priority_vaccines', delimiter=';', scenario=1)
arrival_rate = Utils.arrival_rate(file='arrival_rate', delimiter=';', filter='CALCULATED_RATE')

sus_initial = Utils.probabilities(file='input_probabilities', delimiter=';', parameter='InitialSus', filter='BASE_VALUE')
p_s = Utils.probabilities(file='input_probabilities', delimiter=';', parameter='ExposedToPresymp', filter='BASE_VALUE')
p_c = Utils.probabilities(file='input_probabilities', delimiter=';', parameter='SympToHouse', filter='BASE_VALUE')
p_h = Utils.probabilities(file='input_probabilities', delimiter=';', parameter='SympToHospital', filter='BASE_VALUE')

t_e = Utils.input_time(file='input_time', delimiter=';', parameter='IncubationTime', filter='BASE_VALUE')
t_a = Utils.input_time(file='input_time', delimiter=';', parameter='RecoveryTimeAsymptomatic', filter='BASE_VALUE')
t_p = Utils.input_time(file='input_time', delimiter=';', parameter='PresymptomaticTime', filter='BASE_VALUE')
t_sy = Utils.input_time(file='input_time', delimiter=';', parameter='SymptomsToTreatment', filter='BASE_VALUE')
t_r = Utils.input_time(file='input_time', delimiter=';', parameter='TreatmentToRecovery', filter='BASE_VALUE')
t_d = Utils.input_time(file='input_time', delimiter=';', parameter='TimeToDeath', filter='BASE_VALUE')

contact_matrix = Utils.contact_matrices(file='contact_matrix', delimiter=',')
print('Calculated Vaccination.....')
result_vd = {}
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
                setting.update({'calibration': True, 'arrival_rate': arrival_rate[dept],
                                'age_group': ka, 'work_group': kw, 'health_group': kh,
                                'epsilon_1': 0.0, 'epsilon_2': 0.0, 't_e': t_e['ALL'], 't_a': t_a['ALL'],
                                't_p': t_p['ALL'], 't_sy': t_sy['ALL'], 't_r': t_r['ALL'], 't_d': t_d[ka],
                                'p_s': p_s[ka], 'p_c': p_c[ka], 'p_h': p_h[ka], 'p_i': (1 - (p_c[ka] + p_h[ka])),
                                'total_population': total_population_dept,
                                'population_initial': age_work_health,
                                'priority_vaccine': priority_vaccine,
                                'vaccine_capacities': vaccine_capacities[dept]})
                resp = vaccine.run(days=230, **setting)
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
Utils.save('vaccination_dynamics', result_vd)
