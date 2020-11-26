import datetime
import time
import pandas as pd
import numpy as np
from typing import List
from numpy import double
from logic.compartments import Compartments
from logic.disease_model import DiseaseModel
from logic.utils import Utils
from root import DIR_OUTPUT


class SEIR_US(DiseaseModel):

    def __init__(self, _compartments: List[Compartments], r0: float):
        """
        Initialize the run of contact tracing disease model
        """
        super().__init__(_compartments, r0=r0)
        self.population = Utils.population(file='population', year=2020)
        self._num_comp = len(_compartments)
        self.util = Utils()

    def equations(self, x, t, **kwargs):
        try:
            dx = np.zeros(self._num_comp, dtype=double)
            b = kwargs.get('b') if type(kwargs.get('b')) is float else 1.0
            beta = kwargs.get('beta') if type(kwargs.get('beta')) is float else 1.0
            gamma = kwargs.get('gamma') if type(kwargs.get('gamma')) is float else 1.0
            delta = kwargs.get('delta') if type(kwargs.get('delta')) is float else 1.0
            cm0 = kwargs.get('conmat0') if type(kwargs.get('conmat0')) is float else 1.0
            cm1 = kwargs.get('conmat1') if type(kwargs.get('conmat1')) is float else 1.0
            cm2 = kwargs.get('conmat2') if type(kwargs.get('conmat2')) is float else 1.0
            cm3 = kwargs.get('conmat3') if type(kwargs.get('conmat3')) is float else 1.0
            cm4 = kwargs.get('conmat4') if type(kwargs.get('conmat4')) is float else 1.0
            cm5 = kwargs.get('conmat5') if type(kwargs.get('conmat5')) is float else 1.0
            cm6 = kwargs.get('conmat6') if type(kwargs.get('conmat6')) is float else 1.0
            cm7 = kwargs.get('conmat7') if type(kwargs.get('conmat7')) is float else 1.0
            nco = 3000

            # I all compartments - number of compartments within county
            icn_all_I = []
            for icI in range(nco):
                icn_all_I.append = sum(x[icI * range(16, 24)])

            # equations for all compartments
            for ic in range(nco):
                # number of compartments within county                
                icn_all = []
                for inc in range(31):
                    icn_all.append = [inc]

                # sums
                s0, s1, s2, s3, s4, s5, s6, s7, e0, e1, e2, e3, e4, e5, e6, e7, i0, i1, i2, i3, i4, i5, i6, i7, r0, r1, r2, r3, r4, r5, r6, r7 = \
                x[icn_all]
                nc = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7 + e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7 + i0 + i1 + i2 + i3 + i4 + i5 + i6 + i7 + r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7
                i_all = i0 + i1 + i2 + i3 + i4 + i5 + i6 + i7
                i_all_o = [i0, i1, i2, i3, i4, i5, i6, i7]

                # transtitions between counties
                fromCurrentF2 = np.zeros(nco, dtype=double)
                toCurrentF2 = np.zeros(nco, dtype=double)
                sum_fromCurrentF = np.zeros(nco, dtype=double)
                sum_toCurrentF = np.zeros(nco, dtype=double)

                for i11 in range(nco):
                    fromCurrentF2[i11] = p[nco * ic + i11 - nco]
                    toCurrentF2[i11] = p[nco * ic + i11 + nco * nco - nco]
                    sum_fromCurrentF[i11] = fromCurrentF2[i11]
                    sum_toCurrentF[i11] = toCurrentF2[i11] * icn_all_I[i11]

                # equations
                F1 = ((np.dot(cm0, i_all_o) + sum(sum_toCurrentF) - sum_fromCurrentF) * beta) / nc
                F2 = ((np.dot(cm1, i_all_o) + sum(sum_toCurrentF) - sum_fromCurrentF) * beta) / nc
                F3 = ((np.dot(cm2, i_all_o) + sum(sum_toCurrentF) - sum_fromCurrentF) * beta) / nc
                F4 = ((np.dot(cm3, i_all_o) + sum(sum_toCurrentF) - sum_fromCurrentF) * beta) / nc
                F5 = ((np.dot(cm4, i_all_o) + sum(sum_toCurrentF) - sum_fromCurrentF) * beta) / nc
                F6 = ((np.dot(cm5, i_all_o) + sum(sum_toCurrentF) - sum_fromCurrentF) * beta) / nc
                F6 = ((np.dot(cm6, i_all_o) + sum(sum_toCurrentF) - sum_fromCurrentF) * beta) / nc
                F7 = ((np.dot(cm7, i_all_o) + sum(sum_toCurrentF) - sum_fromCurrentF) * beta) / nc

                s0_dt = {1: s0 * F0, 2: b * n}
                s1_dt = {1: s1 * F1, 2: b * n}
                s2_dt = {1: s2 * F2, 2: b * n}
                s3_dt = {1: s3 * F3, 2: b * n}
                s4_dt = {1: s4 * F4, 2: b * n}
                s5_dt = {1: s5 * F5, 2: b * n}
                s6_dt = {1: s6 * F6, 2: b * n}
                s7_dt = {1: s7 * F7, 2: b * n}
                e0_dt = {1: s0 * F0, 2: gamma * e0}
                e1_dt = {1: s1 * F1, 2: gamma * e1}
                e2_dt = {1: s2 * F2, 2: gamma * e2}
                e3_dt = {1: s3 * F3, 2: gamma * e3}
                e4_dt = {1: s4 * F4, 2: gamma * e4}
                e5_dt = {1: s5 * F5, 2: gamma * e5}
                e6_dt = {1: s6 * F6, 2: gamma * e6}
                e7_dt = {1: s7 * F7, 2: gamma * e7}
                i0_dt = {1: gamma * e0, 2: delta * i0}
                i1_dt = {1: gamma * e1, 2: delta * i1}
                i2_dt = {1: gamma * e2, 2: delta * i2}
                i3_dt = {1: gamma * e3, 2: delta * i3}
                i4_dt = {1: gamma * e4, 2: delta * i4}
                i5_dt = {1: gamma * e5, 2: delta * i5}
                i6_dt = {1: gamma * e6, 2: delta * i6}
                i7_dt = {1: gamma * e7, 2: delta * i7}
                r0_dt = {1: delta * i0}
                r1_dt = {1: delta * i1}
                r2_dt = {1: delta * i2}
                r3_dt = {1: delta * i3}
                r4_dt = {1: delta * i4}
                r5_dt = {1: delta * i5}
                r6_dt = {1: delta * i6}
                r7_dt = {1: delta * i7}

                dx[ic * 0] = -s0_dt[1] + s0_dt[2]
                dx[ic * 1] = -s1_dt[1] + s1_dt[2]
                dx[ic * 2] = -s2_dt[1] + s2_dt[2]
                dx[ic * 3] = -s3_dt[1] + s3_dt[2]
                dx[ic * 4] = -s4_dt[1] + s4_dt[2]
                dx[ic * 5] = -s5_dt[1] + s5_dt[2]
                dx[ic * 6] = -s6_dt[1] + s6_dt[2]
                dx[ic * 7] = -s7_dt[1] + s7_dt[2]
                dx[ic * 8] = e0_dt[1] + e0_dt[2]
                dx[ic * 9] = e1_dt[1] + e1_dt[2]
                dx[ic * 10] = e2_dt[1] + e2_dt[2]
                dx[ic * 11] = e3_dt[1] + e3_dt[2]
                dx[ic * 12] = e4_dt[1] + e4_dt[2]
                dx[ic * 13] = e5_dt[1] + e5_dt[2]
                dx[ic * 14] = e6_dt[1] + e6_dt[2]
                dx[ic * 15] = e7_dt[1] + e7_dt[2]
                dx[ic * 16] = i0_dt[1] + i0_dt[2]
                dx[ic * 17] = i1_dt[1] + i1_dt[2]
                dx[ic * 18] = i2_dt[1] + i2_dt[2]
                dx[ic * 19] = i3_dt[1] + i3_dt[2]
                dx[ic * 20] = i4_dt[1] + i4_dt[2]
                dx[ic * 21] = i5_dt[1] + i5_dt[2]
                dx[ic * 22] = i6_dt[1] + i6_dt[2]
                dx[ic * 23] = i7_dt[1] + i7_dt[2]
                dx[ic * 24] = r0_dt[1]
                dx[ic * 25] = r1_dt[1]
                dx[ic * 26] = r2_dt[1]
                dx[ic * 27] = r3_dt[1]
                dx[ic * 28] = r4_dt[1]
                dx[ic * 29] = r5_dt[1]
                dx[ic * 30] = r6_dt[1]
                dx[ic * 31] = r7_dt[1]

            return dx
        except Exception as e:
            print('Error equations: {0}'.format(e))
            return None


if __name__ == "__main__":
    start_processing_s = time.process_time()
    start_time = datetime.datetime.now()

    compartments = []
    for icl in range(3000):
        s0 = Compartments(name='S0_' + str(icl), value=10000)
        s1 = Compartments(name='S1_' + str(icl), value=10000)
        s2 = Compartments(name='S2_' + str(icl), value=10000)
        s3 = Compartments(name='S3_' + str(icl), value=10000)
        s4 = Compartments(name='S4_' + str(icl), value=10000)
        s5 = Compartments(name='S5_' + str(icl), value=10000)
        s6 = Compartments(name='S6_' + str(icl), value=10000)
        s7 = Compartments(name='S7_' + str(icl), value=10000)

        compartments.append(s0)
        compartments.append(s1)
        compartments.append(s2)
        compartments.append(s3)
        compartments.append(s4)
        compartments.append(s5)
        compartments.append(s6)
        compartments.append(s7)

        e0 = Compartments(name='E0_' + str(icl), value=0)
        e1 = Compartments(name='E1_' + str(icl), value=0)
        e2 = Compartments(name='E2_' + str(icl), value=0)
        e3 = Compartments(name='E3_' + str(icl), value=0)
        e4 = Compartments(name='E4_' + str(icl), value=0)
        e5 = Compartments(name='E5_' + str(icl), value=0)
        e6 = Compartments(name='E6_' + str(icl), value=0)
        e7 = Compartments(name='E7_' + str(icl), value=0)

        compartments.append(e0)
        compartments.append(e1)
        compartments.append(e2)
        compartments.append(e3)
        compartments.append(e4)
        compartments.append(e5)
        compartments.append(e6)
        compartments.append(e7)

        i0 = Compartments(name='I0_' + str(icl), value=0)
        i1 = Compartments(name='I1_' + str(icl), value=0)
        i2 = Compartments(name='I2_' + str(icl), value=0)
        i3 = Compartments(name='I3_' + str(icl), value=0)
        i4 = Compartments(name='I4_' + str(icl), value=1)
        i5 = Compartments(name='I5_' + str(icl), value=0)
        i6 = Compartments(name='I6_' + str(icl), value=0)
        i7 = Compartments(name='I7_' + str(icl), value=0)

        compartments.append(i0)
        compartments.append(i1)
        compartments.append(i2)
        compartments.append(i3)
        compartments.append(i4)
        compartments.append(i5)
        compartments.append(i6)
        compartments.append(i7)

        r0 = Compartments(name='R0_' + str(icl), value=0)
        r1 = Compartments(name='R1_' + str(icl), value=0)
        r2 = Compartments(name='R2_' + str(icl), value=0)
        r3 = Compartments(name='R3_' + str(icl), value=0)
        r4 = Compartments(name='R4_' + str(icl), value=1)
        r5 = Compartments(name='R5_' + str(icl), value=0)
        r6 = Compartments(name='R6_' + str(icl), value=0)
        r7 = Compartments(name='R7_' + str(icl), value=0)

        compartments.append(r0)
        compartments.append(r1)
        compartments.append(r2)
        compartments.append(r3)
        compartments.append(r4)
        compartments.append(r5)
        compartments.append(r6)
        compartments.append(r7)

    ct = SEIR_US(_compartments=compartments, r0=2.0)
    kwargs = {'b': 0.012 / 365, 'beta': 0.5, 'gamma': 1 / 5, 'delta': 1 / 5}
    resp = ct.run(days=365, **kwargs)
    file_csv = DIR_OUTPUT + "{0}.csv".format('seir_us')
    df = pd.DataFrame.from_dict(resp)
    df.to_csv(file_csv)
    end_processing_s = time.process_time()
    end_processing_ns = time.process_time_ns
    end_time = datetime.datetime.now()
    print(df.to_string())
    print('Performance: {0}'.format(end_processing_s - start_processing_s))
    time_diff = (end_time - start_time)
    execution_time = time_diff.total_seconds() * 1000
    print('Execution Time: {0} milliseconds'.format(execution_time))
