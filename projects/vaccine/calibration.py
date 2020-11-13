import pandas as pd
import numpy as np
import os
from root import DIR_INPUT, DIR_OUTPUT
from logic.compartments import Compartments
from logic.utils import Utils

class Calibration(object):

    def __init__(self, case_sim:dict):
        self.case_sim = case_sim
        print('Calibration')

    @staticmethod
    def __obtain_thetas__(x, y):
        try:
            x_matrix = np.column_stack((x ** 2, x, np.ones(x.shape[0])))
            return np.dot(np.linalg.inv(np.dot(np.transpose(x_matrix), x_matrix)), np.dot(np.transpose(x_matrix), y))
        except Exception as e:
            print('Error obtain_thetas: {0}'.format(e))
            return dict()

    @staticmethod
    def __calculate_case__(beta=0.5) -> dict:
        try:

            case_sim = 0 #caso_real+np.random.rand(caso_real.shape[0])*beta-beta*.5 #ACA ES EL CASO REAL
            return case_sim
        except Exception as e:
            print('Error calculate_case: {0}'.format(e))
            return dict()

    @staticmethod
    def run(initial_cases:int = 30, total:bool = True) -> dict:
        fileread = DIR_INPUT + 'real_cases.csv'
        try:
            cases_real = pd.read_csv(fileread, index_col=0, sep=';')
            for c in cases_real.columns:
                cases_real[c] = cases_real[c].apply(lambda a: int(a.replace(',', '')))
            case_real = np.array()
            if total:
                case_real = cases_real['total'].as_matrix()
            else:
                case_real = cases_real['new'].as_matrix()

            x_L = list()
            y_L = list()
            x = np.random.triangular(0, 0.5, 1, size=initial_cases)
            for i in range(initial_cases):
                x_L.append(x[i])
                case_sim = Calibration.__calculate_case__(x[i], case_real)  # Revisar como integrar json acá
                y_L.append(np.sum(np.power(case_sim / case_real - 1, 2)) / case_real.shape[0])
            x = np.array(x_L)
            y = np.array(y_L)
            theta = Calibration.__obtain_thetas__(x, y)
            x_new = -theta[1] / (2 * theta[0])
            case_sim = Calibration.__calculate_case__(x_new, case_real)  # Revisar como integrar json acá
            y_new = np.sum(np.power(case_sim / case_real - 1, 2)) / case_real.shape[0]
            while x_new not in x_L:
                x_L.append(x_new)
                y_L.append(y_new)
                x = np.array(x_L)
                y = np.array(y_L)
                theta = Calibration.__obtain_thetas__(x, y)
                x_new = -theta[1] / (2 * theta[0])
                case_sim = Calibration.__calculate_case__(x_new, case_real)  # Revisar como integrar json acá
                y_new = np.sum(np.power(case_sim / case_real - 1, 2)) / case_real.shape[0]
            x_ideal = min(x_L)
            y_ideal = y_L[x_L.index(x_ideal)]
            return {'beta': x_ideal, 'error': y_ideal}
        except Exception as e:
            print('Error calibration: {0}'.format(e))
            return dict()