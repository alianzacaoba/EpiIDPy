import csv
import datetime
import io
import json
import numpy as np
import pandas as pd
from itertools import groupby
from json import JSONEncoder
from operator import itemgetter
from numpy import double
from root import DIR_INPUT, DIR_OUTPUT


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
                    out.append({str(k).lower().replace(' ', '_'): v for k, v in row.items()})
            f.close()
            group = {}
            for key, _ in groupby(out, key=itemgetter('departamento', 'sigla', str(year))):
                dep = key[0]
                if key[0] not in group:
                    group[dep] = {key[1]: double(key[2])}
                else:
                    val = dict(group[dep])
                    val.update({key[1]: double(key[2])})
                    group[dep] = val
            return group
        except Exception as e:
            print('Error load_population: {0}'.format(e))
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
                data = [row for row in reader if int(row['scenario']) == scenario]
            f.close()
            out = [{'age_group': key[0], 'work_group': key[1], 'health_risk': key[2]}
                   for key, _ in groupby(data, key=itemgetter('age_group', 'work_group', 'health_risk'))]
            return out
        except Exception as e:
            print('Error region_capacities: {0}'.format(e))
            return list()

    @staticmethod
    def contact_matrices(file: str, delimiter: str = ","):
        """Load Contact Matrix file
        :param file: name of file.
        :type file: str
        :param delimiter: delimiter
        :type delimiter: str
        :returns: List
        :rtype: List
        """
        try:
            results = []
            file_path = DIR_INPUT + file + '.csv'
            with open(file_path, 'r', newline='', encoding='utf-8-sig') as f:
                reader = csv.reader(f, delimiter=delimiter)  # change contents to floats
                next(reader)
                for row in reader:  # each row is a list
                    results.append(row)
            f.close()
            return np.array(results, dtype=float)
        except Exception as e:
            print('Error load_contact_matrices: {0}'.format(e))
            return None

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
            date_file = datetime.datetime.now().strftime("%Y-%m-%d_h%Hm%M")
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

            file_csv = DIR_OUTPUT + "{0}_{1}.csv".format(file, date_file)
            with io.open(file_csv, 'w', newline='') as csv_output:
                writer = csv.DictWriter(csv_output, fieldnames=['department', 'age_group', 'compartment', 'result'])
                writer.writeheader()
                for k, v in data.items():
                    for kk, vv in dict(v).items():
                        for kkk, vvv in dict(vv).items():
                            writer.writerow({'department':k, 'age_group':kk, 'compartment':kkk,'result': vvv})
            print('File CSV {0} export successfully!'.format(file_json))
        except Exception as e:
            print('Error save: {0}'.format(e))
            return None
