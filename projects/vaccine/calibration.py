import pandas as pd
import numpy as np
import datetime
from root import DIR_INPUT, DIR_OUTPUT
from logic.compartments import Compartments
from logic.utils import Utils
from projects.vaccine.model_vaccine import ModelVaccine


class Calibration(object):

    def __init__(self, initial_cases: int = 30, total: bool = True):
        self._initial_cases = initial_cases
        self._total = total

    @staticmethod
    def __obtain_thetas__(x, y):
        try:
            x_matrix = np.column_stack((x ** 2, x, np.ones(x.shape[0])))
            return np.dot(np.linalg.inv(np.dot(np.transpose(x_matrix), x_matrix)), np.dot(np.transpose(x_matrix), y))
        except Exception as e:
            print('Error obtain_thetas: {0}'.format(e))
            return dict()

    def run(self) -> dict:
        initial_cases = self._initial_cases
        total = self._total
        file_read = DIR_INPUT + 'real_cases.csv'
        ut = Utils()
        initial_population = ut.initial_population(file='initial_population', delimiter=';')
        total_population = ut.total_population(file='initial_population', delimiter=';')
        vaccine_capacities = ut.region_capacities(file='region_capacities')
        priority_vaccine = ut.priority_vaccine(file='priority_vaccines', delimiter=';', scenario=1)
        arrival_rate = ut.arrival_rate(file='arrival_rate', delimiter=';', filter='SYMPTOMATIC_RATE')

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


        try:
            cases_real = pd.read_csv(file_read, index_col=0, sep=';')
            case_real = np.empty(1)
            if total:
                case_real = cases_real['total'].to_numpy()
            else:
                case_real = cases_real['new'].to_numpy()

            x_L = list()
            y_L = list()
            base = 0.00005
            superior = 0.1
            x = np.random.triangular(0, base, superior, size=initial_cases)
            for i in range(initial_cases):
                print('ITERATION:', i + 1, 'TRAINING BETA', x[i], datetime.datetime.now())
                x_L.append(x[i])
                mv = ModelVaccine()
                kwargs = {'initial_population': initial_population, 'total_population': total_population,
                          'vaccine_capacities': vaccine_capacities, 'priority_vaccine': priority_vaccine,
                          'arrival_rate': arrival_rate, 'sus_initial': sus_initial, 'p_s': p_s, 'p_c': p_c,
                          'p_h': p_h, 'p_dc': p_dc, 'p_dh': p_dh, 'p_di': p_di, 't_e': t_e, 't_a': t_a,
                          't_p': t_p, 't_sy': t_sy, 't_r': t_r, 't_d': t_d, 'beta': float(x[i]), 'calibration': True,
                          'sim_length': 236}
                case_sim = mv.run(**kwargs)['totalCases'][14:]
                del mv
                if not total:
                    case_sim2 = case_sim.copy()
                    for k in range(1, len(case_sim)):
                        case_sim = case_sim2[k] - case_sim2[k - 1]
                    del case_sim2
                y_Act = np.sum(np.power(case_sim / case_real - 1, 2)) / case_real.shape[0]
                y_L.append(y_Act)
                print('RESULT:', y_Act)
            x = np.array(x_L)
            y = np.array(y_L)
            x_new = 0.0
            theta = Calibration.__obtain_thetas__(x, y)
            if theta[0] != 0:
                x_new = -theta[1] / (2 * theta[0])
            else:
                x_new = np.random.triangular(0, base, superior)
            while x_new not in x_L:
                print('ITERATION:', len(x_L) + 1, ' BEST BETA:', x_new, datetime.datetime.now())
                mv = ModelVaccine()
                kwargs = {'initial_population': initial_population, 'total_population': total_population,
                          'vaccine_capacities': vaccine_capacities, 'priority_vaccine': priority_vaccine,
                          'arrival_rate': arrival_rate, 'sus_initial': sus_initial, 'p_s': p_s, 'p_c': p_c,
                          'p_h': p_h, 'p_dc': p_dc, 'p_dh': p_dh, 'p_di': p_di, 't_e': t_e, 't_a': t_a,
                          't_p': t_p, 't_sy': t_sy, 't_r': t_r, 't_d': t_d, 'beta': float(x_new), 'calibration': True,
                          'sim_length': 236}
                case_sim = mv.run(**kwargs)['totalCases'][14:]
                del mv
                if not total:
                    case_sim2 = case_sim.copy()
                    for k in range(1, len(case_sim)):
                        case_sim = case_sim2[k] - case_sim2[k - 1]
                    del case_sim2
                y_new = np.sum(np.power(case_sim / case_real - 1, 2)) / case_real.shape[0]
                print(y_new)
                x_L.append(x_new)
                y_L.append(y_new)
                x = np.array(x_L)
                y = np.array(y_L)
                theta = Calibration.__obtain_thetas__(x, y)
                if theta[0] != 0:
                    x_new = -theta[1] / (2 * theta[0])
                else:
                    x_new = np.random.triangular(0,  base, superior)
            x_ideal = min(x_L)
            y_ideal = y_L[x_L.index(x_ideal)]
            results = {'beta': x_ideal, 'error': y_ideal, 'beta_list': x_L, 'error_list': y_L}
            if total:
                ut.save('calibration_results_total', results)
            else:
                ut.save('calibration_results_new', results)
            return results
        except Exception as e:
            print('Error calibration: {0}'.format(e))
            return dict()


start_time = datetime.datetime.now()
calibration_model = Calibration(initial_cases=40, total=True)
results = calibration_model.run()
end_time = datetime.datetime.now()
time_diff = (end_time - start_time)
execution_time = time_diff.total_seconds()
mm = int(execution_time / 60)
ss = int(execution_time % 60)
for c in results:
    print(c, results[c])
print('Execution Time: {0} minutes {1} seconds'.format(mm, ss))
print('Execution Time: {0} milliseconds'.format(execution_time * 1000))

start_time = datetime.datetime.now()
calibration_model2 = Calibration(initial_cases=40, total=False)
results2 = calibration_model2.run()
for c in results2:
    print(c, results2[c])
end_time = datetime.datetime.now()
time_diff2 = (end_time - start_time)
execution_time2 = time_diff2.total_seconds()
mm = int(execution_time2 / 60)
ss = int(execution_time2 % 60)
print('Execution Time: {0} minutes {1} seconds'.format(mm, ss))
print('Execution Time: {0} milliseconds'.format(execution_time2 * 1000))
