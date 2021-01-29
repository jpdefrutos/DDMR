import numpy as np
import re
import os


def show_and_select(file_list, msg='Select a file by the number: ', int_if_single=True):
    # If the selection is a single number, then return that number instead of the list of length 1
    invalid_selection = True
    while invalid_selection:
        for i, f in enumerate(file_list):
            print('{:03d}) {}'. format(i+1, os.path.split(f)[-1]))

        sel = np.asarray(re.split(',\s|,|\s',input(msg)), np.int) - 1

        if (np.all(sel >= 0)) and (np.all(sel <= len(file_list))):
            invalid_selection = False
    sel = [file_list[s] for s in sel]
    print('Selected: ', ', '.join([os.path.split(f)[-1] for f in sel]))

    if int_if_single:
        if len(sel) == 1:
            sel = sel[0]
    return sel
