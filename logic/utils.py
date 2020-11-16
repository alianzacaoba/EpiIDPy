import csv
import datetime
import io
import json
import operator
import itertools
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
    def population(file: str, delimiter: str = ',', year: int = 2020) -> dict:
        """Load Population file
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
            out = []
            file_path = DIR_INPUT + file + '.csv'
            with open(file_path, newline='', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                for row in reader:
                    out.append({str(k).strip(): v for k, v in row.items()})
            f.close()
            group = {}
            for key, _ in groupby(out, key=operator.itemgetter('DEPARTMENT', 'AGE GROUP', str(year))):
                dep = key[0]
                value = double(key[2].replace(',', ''))
                if key[0] not in group:
                    group[dep] = {key[1]: value}
                else:
                    tmp = dict(group[dep])
                    tmp[key[1]] = value
                    group[dep] = tmp
            return group
        except Exception as e:
            print('Error load_population: {0}'.format(e))
            return dict()

    @staticmethod
    def initial_population(file: str, delimiter: str = ',') -> dict:
        """Load Population file
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
            result = {}
            file_path = DIR_INPUT + file + '.csv'
            with open(file_path, newline='', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                data = [i for i in reader]
                d = sorted(data, key=operator.itemgetter('DEPARTMENT', 'AGE_GROUP', 'WORK_GROUP', 'HEALTH_GROUP'))
                out = {}
                for key, v in groupby(d, key=operator.itemgetter('DEPARTMENT')):
                    if key not in out:
                        out[key] = list(v)
                    else:
                        tmp = list(out[key])
                        out[key] = tmp.extend(v)

                for dept, rows in out.items():
                    age_dict = {}
                    for row in rows:
                        age_group = str(row['AGE_GROUP'])
                        work_group = str(row['WORK_GROUP'])
                        health_group = str(row['HEALTH_GROUP'])
                        population = int(row['POPULATION'])
                        if age_group not in age_dict:
                            age_dict[age_group] = {work_group: {health_group: population}}
                        else:
                            value_work = dict(age_dict[age_group])
                            if work_group not in value_work:
                                value_work[work_group] = {health_group: population}
                            else:
                                value_heath = value_work[work_group]
                                if health_group not in value_heath:
                                    value_heath[health_group] = population
                                value_work[work_group] = value_heath
                            age_dict[age_group] = value_work
                    result[dept] = age_dict
                # print(result)
            f.close()
            return result
        except Exception as e:
            print('Error initial_population: {0}'.format(e))
            return dict()

    @staticmethod
    def total_population(file: str, delimiter: str = ',') -> dict:
        """Load Population file
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
            result = {}
            file_path = DIR_INPUT + file + '.csv'
            with open(file_path, newline='', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                data = [i for i in reader]
                d = sorted(data, key=operator.itemgetter('DEPARTMENT', 'AGE_GROUP', 'WORK_GROUP', 'HEALTH_GROUP'))
                out = {}
                for key, v in groupby(d, key=operator.itemgetter('DEPARTMENT')):
                    if key not in out:
                        out[key] = list(v)
                    else:
                        tmp = list(out[key])
                        out[key] = tmp.extend(v)
                for dept, rows in out.items():
                    total = 0.0
                    for row in rows:
                        population = float(row['POPULATION'])
                        total += population
                    result[dept] = total
            f.close()
            return result
        except Exception as e:
            print('Error initial_population: {0}'.format(e))
            return dict()

    @staticmethod
    def probabilities(file: str, delimiter: str = ',', parameter_1: str = 'InitialSus',
                      parameter_2: str = 'None', filter: str = 'BASE_VALUE') -> dict:
        """Load probabilities file
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
                    key = row['AGE_GROUP']
                    parameter = str(row['PARAMETER'])
                    if parameter == parameter_1 and parameter_2 == 'ALL':
                        out[key] = float(row[filter])
                    elif parameter == parameter_1 and parameter_2 == 'None':
                        health = row['HEALTH_GROUP']
                        if key not in out:
                            out[key] = {health: float(row[filter])}
                        else:
                            tmp_value = dict(out[key])
                            tmp_value[health] = float(row[filter])
                            out[key] = tmp_value
            f.close()
            return out
        except Exception as e:
            print('Error probabilities: {0}'.format(e))
            return dict()

    @staticmethod
    def input_time(file: str, delimiter: str = ',', parameter: str = 'InitialSus', filter: str = 'BASE_VALUE') -> dict:
        """Load input_time file
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
                    if str(row['PARAMETER']) == parameter:
                        out[row['AGE_GROUP']] = float(row[filter])
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
    def priority_vaccine(file: str, delimiter: str = ",", scenario: int = 1) -> list:
        """Load priority age group by department.
        :param file: name of file.
        :type file: str
        :param delimiter: delimiter
        :type delimiter: str
        :returns: list of priority .
        :rtype: list
        """
        try:
            file_path = DIR_INPUT + file + '.csv'
            with open(file_path, newline='', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                data = []
                for row in reader:
                    value = int(row['SCENARIO'])
                    if value == scenario:
                        data.append(row)
            f.close()
            out = [{'AGE_GROUP': key[0], 'WORK_GROUP': key[1], 'HEALTH_GROUP': key[2]}
                   for key, _ in groupby(data, key=operator.itemgetter('AGE_GROUP', 'WORK_GROUP', 'HEALTH_GROUP'))]
            return out
        except Exception as e:
            print('Error priority_vaccine: {0}'.format(e))
            return list()

    @staticmethod
    def contact_matrices(file: str, delimiter: str = ",") -> dict:
        """Load Contact Matrix file
        :param file: name of file.
        :type file: str
        :param delimiter: delimiter
        :type delimiter: str
        :returns: List
        :rtype: List
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
    def region_capacities(file: str, delimiter: str = ",") -> dict:
        """Load region capacities
        :param file: name of file.
        :type file: str
        :param delimiter: delimiter
        :type delimiter: str
        :returns: Dict of region capacities
        :rtype: dict
        """
        try:
            out = {}
            file_path = DIR_INPUT + file + '.csv'
            with open(file_path, newline='', encoding='utf-8-sig') as f:
                reader = csv.reader(f, delimiter=delimiter)
                next(reader)
                for row in reader:
                    out[row[0]] = int(row[1])
            f.close()
            return out
        except Exception as e:
            print('Error region_capacities: {0}'.format(e))
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
            Utils.export_excel(file=file, data=data)
        except Exception as e:
            print('Error save: {0}'.format(e))
            return None

    @staticmethod
    def export_excel(file: str, data: dict):
        # Write XLSX file
        date_file = datetime.datetime.now().strftime(DATE_FORMAT)

        print('Generating excel report....')
        result = {}
        for dept, age_work_health in tqdm(data.items()):
            position = int(str(dept).find(" "))
            dept = str(dept)[: len(dept) if str(dept).find(" ") == -1 else (position + 1)].lower()
            file_xlsx = DIR_REPORT + '{0}_{1}_{2}.xlsx'.format(file, dept, date_file)
            with pd.ExcelWriter(file_xlsx) as writer_xls:
                for ka, va in dict(age_work_health).items():
                    for kw, vw in dict(va).items():
                        for kh, vh in dict(vw).items():
                            sheet_name = '{0}_{1}_{2}'.format(ka, kw, kh)
                            df = pd.DataFrame(vh)
                            df.to_excel(writer_xls, sheet_name=sheet_name)
            writer_xls.close()
        print('Files XLSX {0} export successfully!')
