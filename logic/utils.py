import csv
import io
import numpy
import json
import datetime
import numpy as np
from itertools import groupby
from operator import itemgetter
from numpy import double
from json import JSONEncoder
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

    def __init__(self):
        print('Utils class')

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
    def vaccine_population(year: int = 2020, scenario: int = 1, work_group: str = 'M', wr_percent: float = 0.1) -> dict:
        """Load vaccine Population
        :param year: year of population
        :type year: int
        :param wr_percent: risk group percentage
        :type wr_percent: float
        :param scenario: scenario to evaluate
        :type scenario: int
        :returns: dict vaccine population
        :rtype: dict
        """
        try:
            population = Utils.population(file='population', year=year)
            priority_vaccine = Utils.priority(file='priority')
            age_priority = {row['age_group']: float(row['percent']) for row in priority_vaccine
                            if int(row['scenario']) == scenario and row['work_group'] == work_group}
            out = {}
            for dep, age_group in population.items():
                out[dep] = {k: int((age_group[k] * wr_percent) * v) for k, v in age_priority.items()}
            return out
        except Exception as e:
            print('Error vaccine_population: {0}'.format(e))
            return dict()

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
    def priority(file: str, delimiter: str = ",") -> list:
        """Load priority age group by department.
        :param file: name of file.
        :type file: str
        :param delimiter: delimiter
        :type delimiter: str
        :returns: list of priority age group by department.
        :rtype: list
        """
        try:
            out = []
            file_path = DIR_INPUT + file + '.csv'
            with open(file_path, newline='', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                for row in reader:
                    out.append({str(k): v for k, v in row.items()})
            f.close()
            return out
        except Exception as e:
            print('Error region_capacities: {0}'.format(e))
            return list()

    @staticmethod
    def save_json(file: str, data: dict):
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
            file_path = DIR_OUTPUT + "{0}_{1}.json".format(file, date_file)
            # Write JSON file
            with io.open(file_path, 'w', encoding='utf8') as outfile:
                str_ = json.dumps(data,
                                  indent=4, sort_keys=True,
                                  separators=(',', ': '),
                                  ensure_ascii=False,
                                  cls=NumpyArrayEncoder)
                outfile.write(str_)
            outfile.close()
            print('File {0} export successfully!'.format(file_path))
        except Exception as e:
            print('Error save_json: {0}'.format(e))
            return None

    @staticmethod
    def save_scv(file: str, data: dict):
        """Save CSV file
        :param file: name of file.
        :type file: str
        :param data: Dict of data.
        :type data: dict
        :returns: CSV file
        :rtype: CSV file
        """
        try:
            date_file = datetime.datetime.now().strftime("%Y-%m-%d_h%Hm%M")
            file_path = DIR_OUTPUT + "{0}_{1}.csv".format(file, date_file)
            with open(file_path, 'w', encoding='utf8') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=['work_risk', 'age_groups', 'compartments'])
                for key, val in data.items():
                    row = {}
                    for k, value in val.items():
                        for key_age, v in dict(value).items():
                            row['work_risk'] = key
                            row['age_groups'] = key_age
                            row['compartments'] = v
                            writer.writerow(row)
            outfile.close()
            print('File {0} export successfully!'.format(file_path))
        except IOError as e:
            print('Error save_scv: {0}'.format(e))
            return None
