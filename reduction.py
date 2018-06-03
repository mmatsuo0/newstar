#!/usr/bin/env python
# Python 2.7.14

import argparse
import os
import struct
import matplotlib.pyplot
import numpy
import scipy.optimize
import copy
import glob


class ReductionNewstarData:
    def __init__(self, file_path):
        self.file_path = file_path
        self.param_dict = {}
        self.param_dict_tmp = {}
        self.data = ''
        self.param_dict['data_name'] = os.path.basename(file_path)
        self.load_data()
        self.analysis_column()
        self.make_data_array()
        self.get_backend_info()

    def load_data(self):
        with open(self.file_path, 'r') as f1:
            raw_data = f1.read()
        header, data = raw_data.split('END ')
        n_params = len(header) / 80
        for line in range(n_params):
            p = header[line*80: (line+1)*80]
            if p.find('=') == -1:
                continue
            param = p.split('/')
            key, val = param[0].split('=', 1)
            self.param_dict[key.strip()] = val.strip().strip('\'').strip()
        self.data = data.split('LS  ')

    def analysis_column(self):
        sbyte = 0
        for column in range(int(self.param_dict['TFIELDS'])-1):
            column += 2
            self.param_dict['sbyte{}'.format(column)] = sbyte
            tform = self.param_dict['TFORM{}'.format(column)]
            if tform[-1] in ['L', 'B', 'A']:
                self.param_dict['nbyte{}'.format(column)] = 1 * int(tform[:-1])
                self.param_dict['pform{}'.format(column)] = tform[:-1] + 'c'
            elif tform[-1] is 'I':
                self.param_dict['nbyte{}'.format(column)] = 2 * int(tform[:-1])
                # param_dict['pform{}'.format(column)] = 'h'
            elif tform[-1] in ['J', 'E']:
                self.param_dict['nbyte{}'.format(column)] = 4 * int(tform[:-1])
                self.param_dict['pform{}'.format(column)] = tform[:-1] + 'i'
            elif tform[-1] in ['D', 'C', 'P']:
                self.param_dict['nbyte{}'.format(column)] = 8 * int(tform[:-1])
                self.param_dict['pform{}'.format(column)] = tform[:-1] + 'd'
            elif tform[-1] is 'M':
                self.param_dict['nbyte{}'.format(column)] = 16 * int(tform[:-1])
            sbyte += self.param_dict['nbyte{}'.format(column)]
        self.param_dict['nscan'] = (int(self.param_dict['NAXIS2']) /
                                    int(self.param_dict['ARYNM']) - 1)
        self.param_dict_tmp = copy.copy(self.param_dict)

    def get_column_data(self, data_n, column_value):
        p = self.param_dict_tmp
        d = self.data
        c = p.keys()[p.values().index(column_value)].strip('TTYPE')
        c_d = struct.unpack('>{}'.format(p['pform{}'.format(c)]),
                            d[data_n][p['sbyte{}'.format(c)]:
                                      p['sbyte{}'.format(c)]+p['nbyte{}'.format(c)]])
        return c_d

    def make_data_array(self):
        arry3 = self.param_dict['ARRY3']
        arry4 = self.param_dict['ARRY4']
        if '1' in arry3[-4:] or '1' in arry4:
            self.param_dict['spw_mode'] = True
            self.param_dict['n_channel'] = 2048
        else:
            self.param_dict['spw_mode'] = False
            self.param_dict['n_channel'] = 4096
        array_list = []
        for scan_n in range(self.param_dict['nscan']):
            for array_n in range(int(self.param_dict['ARYNM'])):
                data_n = (scan_n + 1) * int(self.param_dict['ARYNM']) + array_n + 1
                array = self.get_column_data(data_n, 'ARRYT')
                scan = self.get_column_data(data_n, 'ISCN')
                data_array = self.get_column_data(data_n, 'LDATA')
                sfctr = self.get_column_data(data_n, 'SFCTR')
                adoff = self.get_column_data(data_n, 'ADOFF')
                array = ''.join(array).strip()
                array = 'A{:02d}'.format(int(array[1:]))
                data_key = 'SCAN{}'.format(scan[0]) + array
                self.param_dict[data_key] = numpy.array(data_array) * sfctr + adoff
                if self.param_dict['spw_mode']:
                    self.param_dict[data_key] = self.param_dict[data_key][:2048]
                array_list.append(array)
        self.param_dict['array_list'] = list(set(array_list))

    def get_backend_info(self):
        for array_n in range(int(self.param_dict['ARYNM'])):
            data_n = int(self.param_dict['ARYNM']) + array_n + 1
            array = self.get_column_data(data_n, 'ARRYT')
            fqtrk = self.get_column_data(data_n, 'FQTRK')
            fqif1 = self.get_column_data(data_n, 'FQIF1')
            f0cal = self.get_column_data(data_n, 'F0CAL')
            fqcal = self.get_column_data(data_n, 'FQCAL')
            chcal = self.get_column_data(data_n, 'CHCAL')
            cwcal = self.get_column_data(data_n, 'CWCAL')
            frq0 = self.get_column_data(data_n, 'FRQ0')
            bebw = self.get_column_data(data_n, 'BEBW')
            beres = self.get_column_data(data_n, 'BERES')
            chwid = self.get_column_data(data_n, 'CHWID')
            sidbd = self.get_column_data(data_n, 'SIDBD')
            # tsys = self.get_column_data(data_n, 'TSYS')
            # print tsys
            # el = self.get_column_data(data_n, 'EL')
            # print el
            rx = self.get_column_data(data_n, 'RX')
            array = ''.join(array).strip()
            array = 'A{:02d}'.format(int(array[1:]))
            rx = ''.join(rx).strip()
            if rx == 'H20ch1':
                rx = 'H22R'
            elif rx == 'H20ch2':
                rx = 'H22L'
            elif 'TMULT' in rx:
                rx = rx.replace('TMULT', 'FOREST')
            self.param_dict['{}_rx'.format(array)] = rx
            sidbd = ''.join(sidbd).strip()
            self.param_dict['{}_sidbd'.format(array)] = sidbd
            self.param_dict['{}_fqtrk'.format(array)] = fqtrk
            self.param_dict['{}_fqif1'.format(array)] = fqif1
            self.param_dict['{}_f0cal'.format(array)] = f0cal
            self.param_dict['{}_fqcal'.format(array)] = fqcal
            self.param_dict['{}_chcal'.format(array)] = chcal
            self.param_dict['{}_cwcal'.format(array)] = cwcal
            self.param_dict['{}_frq0'.format(array)] = frq0
            self.param_dict['{}_bebw'.format(array)] = bebw
            self.param_dict['{}_beres'.format(array)] = beres
            self.param_dict['{}_chwid'.format(array)] = chwid

    def integrate(self, array):
        data_int = numpy.zeros(self.param_dict['n_channel'])
        for scan_n in range(1, self.param_dict['nscan'] + 1):
            data_int += self.param_dict['SCAN{0}{1}'.format(scan_n, array)]
        data_int /= self.param_dict['nscan']
        return data_int

    def exec_integrate(self):
        for array in sorted(self.param_dict['array_list']):
            self.param_dict['{}_int'.format(array)] = self.integrate(array)

    @staticmethod
    def fitting(parameter, xt, yt):
        at, bt = parameter
        residual = yt - (at * xt + bt)
        return residual

    def base(self, array):
        data_int = self.param_dict['{}_int'.format(array)]
        if self.param_dict['spw_mode']:
            baserange = [100, 500, 1600, 2000]
        else:
            baserange = [100, 500, 3600, 4000]
        x = numpy.arange(1, self.param_dict['n_channel']+1, 1)

        base_x1 = x[numpy.where((x >= baserange[0]) & (x <= baserange[1]))]
        base_x2 = x[numpy.where((x >= baserange[2]) & (x <= baserange[3]))]
        base_y1 = data_int[numpy.where((x >= baserange[0]) & (x <= baserange[1]))]
        base_y2 = data_int[numpy.where((x >= baserange[0]) & (x <= baserange[1]))]
        base_x = numpy.r_[base_x1, base_x2]
        base_y = numpy.r_[base_y1, base_y2]

        params0 = [0, 0]
        params1 = scipy.optimize.leastsq(self.fitting, params0, args=(base_x, base_y))
        af, bf = params1[0]
        basefit = af * x + bf
        result = data_int - basefit
        return result

    def exec_base(self):
        for array in sorted(self.param_dict['array_list']):
            self.param_dict['{}_base'.format(array)] = self.base(array)

    def binning(self, array):
        data = self.param_dict['{}_base'.format(array)]
        binning_ch = self.param_dict['binning_channel']
        data_binning = numpy.zeros(self.param_dict['n_channel'])
        for i in range(binning_ch):
            data_binning += data[i::binning_ch]
        data_binning /= binning_ch
        return data_binning

    def exec_binning(self):
        binning_ch = 1
        self.param_dict['n_channel'] /= binning_ch
        self.param_dict['binning_channel'] = binning_ch
        for array in sorted(self.param_dict['array_list']):
            self.param_dict['{}_data'.format(array)] = self.binning(array)


def main(data_path):
    # Reduction
    print data_path
    reduction_data = ReductionNewstarData(data_path)
    reduction_data.exec_integrate()
    reduction_data.exec_base()
    reduction_data.exec_binning()

    # params = reduction_data.param_dict
    # print params['A01_data']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    args = parser.parse_args()
    main(args.data_path)
