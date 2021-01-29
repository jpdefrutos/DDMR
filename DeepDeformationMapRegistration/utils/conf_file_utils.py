import DeepDeformationMapRegistration.utils.constants as C
import re
import os


class ConfigurationFile:
    def __init__(self,
                 file_path: str):
        self.__file = file_path
        self.__load_configuration()

    def __load_configuration(self):
        fd = open(self.__file, 'r')
        file_lines = fd.readlines()

        for line in file_lines:
            if '#' not in line and line != '\n':
                match = re.match('(.*)=(.*)', line)
                if match[1] in C.__dict__.keys():
                    # Careful with eval!!
                    try:
                        new_val = eval(match[2])
                    except NameError:
                        new_val = match[2]
                    old = C.__dict__[match[1]]
                    C.__dict__[match[1]] = new_val

                    # Special case
                    if match[1] == 'GPU_NUM':
                        C.__dict__[match[1]] = str(new_val)
                        os.environ['CUDA_VISIBLE_DEVICES'] = C.GPU_NUM

                    if match[1] == 'EPOCHS':
                        C.__dict__[match[1]] = new_val
                        C.__dict__['SAVE_EPOCH'] = new_val // 10
                        C.__dict__['VERBOSE_EPOCH'] = new_val // 10

                    if match[1] == 'SAVE_EPOCH' or match[1] == 'VERBOSE_EPOCH':
                        if new_val is not None:
                            C.__dict__[match[1]] = C.__dict__['EPOCHS'] // new_val
                        else:
                            C.__dict__[match[1]] = None

                    if match[1] == 'VALIDATION_ERR_LIMIT_COUNTER':
                        C.__dict__[match[1]] = new_val
                        C.__dict__['VALIDATION_ERR_LIMIT_COUNTER_BACKUP'] = new_val


                    print('INFO: Updating constant {}: {} -> {}'.format(match[1], old, C.__dict__[match[1]]))
                else:
                    print('ERROR: Unknown constant {}'.format(match[1]))

