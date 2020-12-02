import csv
import datetime
import io
import json
import operator
import itertools
import os

from tqdm import tqdm
import numpy as np
import pandas as pd
from itertools import groupby
from json import JSONEncoder
from numpy import double
from root import DIR_INPUT, DIR_OUTPUT, DIR_REPORT

DATE_FORMAT = "%Y-%m-%d_h%Hm%M"


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return float(obj)
        elif isinstance(obj, double):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)


class Utils(object):
    """Class used to support tasks, activities, and process """

    @staticmethod
    def load_json(file: str):
        """Load json file
        :param file: name of file.
        :type file: str
        :returns: JSON object
        :rtype: JSON
        """
        try:
            result = []
            path = DIR_INPUT + file
            f = open(path, "r")
            # Reading from file
            data = json.loads(f.read())
            for row in data:
                result.append(row)
            # Closing file
            f.close()
            return result
        except Exception as e:
            print('Error load_json: {0}'.format(e))
            return None

    @staticmethod
    def initial_population(file: str, delimiter: str = ',') -> dict:
        """Load population by department and work group.
        :param file: name of file.
        :type file: str
        :param delimiter: delimiter
        :type delimiter: str
        :returns: Dictionary of population by department and work group.
        :rtype: dict
        """
        try:
            result = {}
            file_path = DIR_INPUT + file + '.csv'
            work_health_age_dict = {}
            with open(file_path, newline='', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                data = [i for i in reader]
                d = sorted(data, key=operator.itemgetter('DEPARTMENT', 'WORK_GROUP', 'HEALTH_GROUP', 'AGE_GROUP'))
                for key, v in groupby(d, key=operator.itemgetter('DEPARTMENT')):
                    if key not in work_health_age_dict:
                        work_health_age_dict[key] = list(v)
                    else:
                        tmp = list(work_health_age_dict[key])
                        work_health_age_dict[key] = tmp.extend(v)
            f.close()
            for dept, rows in work_health_age_dict.items():
                work_health_dict = {}
                for row in rows:
                    age_group = str(row['AGE_GROUP'])
                    work_group = str(row['WORK_GROUP'])
                    health_group = str(row['HEALTH_GROUP'])
                    population = int(row['POPULATION'])
                    if work_group not in work_health_dict:
                        work_health_dict[work_group] = {health_group: {age_group: population}}
                    else:
                        value_work = dict(work_health_dict[work_group])
                        if health_group not in value_work:
                            value_work[health_group] = {age_group: population}
                        else:
                            value_health = dict(value_work[health_group])
                            if age_group not in value_health:
                                value_health[age_group] = population

                            value_work[health_group] = value_health
                        work_health_dict[work_group] = value_work
                    result[dept] = work_health_dict
            return result
        except Exception as e:
            print('Error initial_population: {0}'.format(e))
            return dict()

    @staticmethod
    def initial_population_ct(file: str, delimiter: str = ',') -> dict:
        """Load population by department and work group.
        :param file: name of file.
        :type file: str
        :param delimiter: delimiter
        :type delimiter: str
        :returns: Dictionary of population by department and work group.
        :rtype: dict
        """
        try:
            result = {}
            file_path = DIR_INPUT + file + '.csv'
            work_health_age_dict = {}
            with open(file_path, newline='', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                data = [i for i in reader]
                d = sorted(data, key=operator.itemgetter('DEPARTMENT', 'WORK_GROUP', 'HEALTH_GROUP', 'AGE_GROUP'))
                for key, v in groupby(d, key=operator.itemgetter('DEPARTMENT')):
                    if key not in work_health_age_dict:
                        work_health_age_dict[key] = list(v)
                    else:
                        tmp = list(work_health_age_dict[key])
                        work_health_age_dict[key] = tmp.extend(v)
            f.close()
            for dept, rows in work_health_age_dict.items():
                work_dict = {}
                for row in rows:
                    age = str(row['AGE_GROUP'])
                    work = str(row['WORK_GROUP'])
                    population = int(row['POPULATION'])
                    if work not in work_dict:
                        work_dict[work] = {age: population}
                    else:
                        age_dict = dict(work_dict[work])
                        if age not in age_dict:
                            age_dict[age] = population
                        else:
                            value_age = age_dict[age]
                            age_dict[age] = value_age + population

                        work_dict[work] = age_dict
                result[dept] = work_dict
            return result
        except Exception as e:
            print('Error initial_population: {0}'.format(e))
            return dict()

    @staticmethod
    def total_population(file: str, delimiter: str = ',') -> dict:
        """Load total population by department.
        :param file: name of file.
        :type file: str
        :param delimiter: delimiter
        :type delimiter: str
        :returns: Dictionary of total population by department.
        :rtype: dict
        """
        try:
            file_path = DIR_INPUT + file + '.csv'
            work_health_age_dict = {}
            with open(file_path, newline='', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                data = [i for i in reader]
                d = sorted(data, key=operator.itemgetter('DEPARTMENT', 'WORK_GROUP', 'HEALTH_GROUP', 'AGE_GROUP'))
                for key, v in groupby(d, key=operator.itemgetter('DEPARTMENT')):
                    if key not in work_health_age_dict:
                        work_health_age_dict[key] = list(v)
                    else:
                        tmp = list(work_health_age_dict[key])
                        work_health_age_dict[key] = tmp.extend(v)
            f.close()
            result = {}
            for dept, rows in work_health_age_dict.items():
                tmp_dict = {}
                for row in rows:
                    age_group = str(row['AGE_GROUP'])
                    population = int(row['POPULATION'])
                    if age_group not in tmp_dict:
                        tmp_dict[age_group] = population
                    else:
                        value = tmp_dict[age_group]
                        tmp_dict[age_group] = value + population
                result[dept] = tmp_dict
            return result
        except Exception as e:
            print('Error initial_population: {0}'.format(e))
            return dict()

    @staticmethod
    def probabilities(delimiter: str = ',', parameter_1: str = 'PresymptomaticInfectionInExposed',
                      parameter_2: str = 'None', filter: str = 'VALUE') -> dict:
        """Load probabilities rate.
        :param file: name of file.
        :type file: str
        :param delimiter: delimiter
        :type delimiter: str
        :param year: year of population
        :type year: int
        :returns: Dictionary of probabilities rate
        :rtype: dict
        """
        try:
            out = {}
            file_path = DIR_INPUT + 'input_probabilities.csv'
            with open(file_path, newline='', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                for row in reader:
                    parameter = str(row['PARAMETER']).strip()
                    if parameter == parameter_1 and parameter_2 == 'ALL':
                        out[row['AGE_GROUP']] = float(row[filter])
                    elif parameter == parameter_1 and parameter_2 == 'None':
                        work = row['WORK_GROUP']
                        age_group = row['AGE_GROUP']
                        if work not in out:
                            out[work] = {age_group: float(row[filter])}
                        else:
                            tmp_value = dict(out[work])
                            if age_group not in tmp_value:
                                tmp_value[age_group] = float(row[filter])
                            out[work] = tmp_value
            f.close()
            return out
        except Exception as e:
            print('Error probabilities: {0}'.format(e))
            return dict()

    @staticmethod
    def input_time(delimiter: str = ',') -> dict:
        """Load input time
        :param file: name of file.
        :type file: str
        :param delimiter: delimiter
        :type delimiter: str
        :returns: Dictionary of population by department.
        :rtype: dict
        """
        try:
            out = {}
            file_path = DIR_INPUT + 'input_time.csv'
            with open(file_path, newline='', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                for row in reader:
                    parameter = str(row['PARAMETER']).strip()
                    value = float(row['VALUE'])
                    if value > 0.0:
                        out[parameter] = 1 / value
                    else:
                        out[parameter] = 0.0
            f.close()
            return out
        except Exception as e:
            print('Error probabilities: {0}'.format(e))
            return dict()

    @staticmethod
    def arrival_rate(file: str, delimiter: str = ',', filter: str = 'CALCULATED_RATE') -> dict:
        """Load arriva rate file
        :param file: name of file.
        :type file: str
        :param delimiter: delimiter
        :type delimiter: str
        :param year: year of population
        :type year: int
        :returns: Dictionary of population by department.
        :rtype: dict
        """
        try:
            out = {}
            file_path = DIR_INPUT + file + '.csv'
            with open(file_path, newline='', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                for row in reader:
                    out[row['DEPARTMENT']] = float(str(row[filter]).replace(',', '.'))
            f.close()
            return out
        except Exception as e:
            print('Error arrival_rate: {0}'.format(e))
            return dict()

    @staticmethod
    def contact_matrices(file: str, delimiter: str = ",") -> dict:
        """Load Contact Matrix file
        :param file: name of file.
        :type file: str
        :param delimiter: delimiter
        :type delimiter: str
        :returns: Dictionary of contact matrix by age
        :rtype: dict
        """
        try:
            out = {}
            file_path = DIR_INPUT + file + '.csv'
            with open(file_path, 'r', newline='', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f, delimiter=delimiter)  # change contents to floats
                for row in reader:  # each row is a list
                    for k, v in dict(row).items():
                        value = double(str(v).strip())
                        if k not in out:
                            out[k] = [value]
                        else:
                            tmp_list = list(out[k]) + [value]
                            out[k] = tmp_list
            f.close()
            return out
        except Exception as e:
            print('Error load_contact_matrices: {0}'.format(e))
            return dict()

    @staticmethod
    def save(file: str, data: dict) -> None:
        """Save json file
        :param file: name of file.
        :type file: str
        :param data: Dict of data
        :type data: dict
        :returns: JSON file
        :rtype: JSON file
        """
        try:
            date_file = datetime.datetime.now().strftime(DATE_FORMAT)
            file_json = DIR_OUTPUT + "{0}_{1}.json".format(file, date_file)
            # Write JSON file
            with io.open(file_json, 'w', encoding='utf8') as json_output:
                str_ = json.dumps(data,
                                  indent=4, sort_keys=True,
                                  separators=(',', ': '),
                                  ensure_ascii=False,
                                  cls=NumpyArrayEncoder)
                json_output.write(str_)
            json_output.close()
            print('File JSON {0} export successfully!'.format(file_json))
        except Exception as e:
            print('Error save: {0}'.format(e))
            return None

    @staticmethod
    def export_excel_ct(file: str, data: dict):
        # Write XLSX file
        if not os.path.exists(DIR_REPORT):
            os.makedirs(DIR_REPORT)
        date_file = datetime.datetime.now().strftime(DATE_FORMAT)
        print('Generating excel report....')
        for dept, work_age in tqdm(data.items()):
            position = int(str(dept).find(" "))
            dept = str(dept)[: len(dept) if str(dept).find(" ") == -1 else (position + 1)].lower()
            file_xlsx = DIR_REPORT + '{0}_{1}_{2}.xlsx'.format(file, dept, date_file)
            with pd.ExcelWriter(file_xlsx) as writer_xls:
                for kw, vw in dict(work_age).items():
                    for age, value in dict(vw).items():
                        sheet_name = '{0}_{1}'.format(age, kw)
                        df = pd.DataFrame(value)
                        df.to_excel(writer_xls, sheet_name=sheet_name)
            writer_xls.close()
        print('Files XLSX {0} export successfully!')
