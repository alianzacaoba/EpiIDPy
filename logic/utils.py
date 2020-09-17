import csv
import io
import sys
import json
import datetime
from operator import itemgetter

import numpy as np
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
    def save_json(file: str, data: dict):
        """Save json file
        :param file: name of file.
        :type file: str
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

