"""Dataset registrations."""
import os

import numpy as np

import common

def LoadDmv(filename='Vehicle__Snowmobile__and_Boat_Registrations.csv'):
    csv_file = './datasets/{}'.format(filename)
    cols = [
        'Record Type','Registration Class', 'State', 'County', 'Body Type',
        'Fuel Type', 'Reg Valid Date', 'Color', 'Scofflaw Indicator',
        'Suspension Indicator', 'Revocation Indicator'
    ]
    # Note: other columns are converted to objects/strings automatically.  We
    # don't need to specify a type-cast for those because the desired order
    # there is the same as the default str-ordering (lexicographical).
    type_casts = {'Reg Valid Date': np.datetime64}
    return common.CsvTable('DMV', csv_file, cols, type_casts)

def LoadCensus(filename='census.csv'):
    csv_file = './datasets/{}'.format(filename)
    cols =[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    type_casts = {}
    return common.CsvTable('Adult', csv_file, cols, type_casts, header=None)


def LoadCup98(filename='cup98.csv'):
    csv_file = './datasets/{}'.format(filename)
    cols = [473, 5, 9, 10, 11, 12, 52, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 74, 6, 17, 18, 361, 19, 20, 21, 22, 42, 73, 470, 472, 475, 477, 50, 476, 478, 318, 25, 51, 54, 408, 474, 358, 384, 14, 317, 326, 350, 161, 234, 148, 324, 13, 147, 88, 299, 316, 397, 410, 288, 233, 258, 183, 325, 387, 182, 307, 357, 314, 322, 92, 304, 323, 210, 93, 464, 94, 255, 271, 15, 219, 259, 3, 294, 359, 96, 209, 336, 265, 319, 360, 272, 277]
    type_casts = {}
    return common.CsvTable('Cup98', csv_file, cols, type_casts, header=None)
