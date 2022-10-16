import warnings

import pandas as pd
import os
import argparse
import re
import shutil

DICT_MODEL_NAMES = {'BASELINE': 'BL',
                    'SEGGUIDED': 'SG',
                    'UW': 'UW'}

DICT_METRICS_NAMES = {'NCC': 'N',
                      'SSIM': 'S',
                      'DICE': 'D',
                      'DICE MACRO': 'D',
                      'HD': 'H', }


def row_name(in_path: str):
    model = re.search('((UW|SEGGUIDED|BASELINE).*)_\d', in_path).group(1).rstrip('_')
    model = model.replace('_Lsim', '')
    model = model.replace('_Lseg', '')
    model = model.replace('_L', '')
    model = model.replace('_', ' ')
    model = model.upper()
    elements = model.split()
    model = elements[0]
    metrics = list()
    model = DICT_MODEL_NAMES[model]
    for m in elements[1:]:
        if m != 'MACRO':
            metrics.append(DICT_METRICS_NAMES[m])

    return '{}-{}'.format(model, ''.join(metrics))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dir', nargs='+', type=str, help='List of directories where metrics.csv file is',
                        default=None)
    parser.add_argument('-o', '--output', type=str, help='Output directory', default=os.getcwd())
    parser.add_argument('--overwrite', type=bool, default=True)
    parser.add_argument('--filename', type=str, help='Output file name', default='metrics')
    parser.add_argument('--removemetrics', nargs='+', type=str, default=None)
    args = parser.parse_args()
    assert args.dir is not None, "No directories provided. Stopping"

    if len(args.dir) == 1:
        list_files = list()
        for r, d, f in os.walk(args.dir[0]):
            for name in f:
                if 'metrics.csv' == name:  # and os.path.split(r)[1] == 'Evaluation_paper':
                    list_files.append(os.path.join(r, name))
    else:
        list_files = [os.path.join(d, 'metrics.csv') for d in args.dir]

        for d in list_files:
            assert os.path.exists(d), "Missing metrics.csv file in: " + os.path.split(d)[0]

    print('Metric files found: {}'.format(list_files))

    dataframes = list()
    if len(list_files):
        for d in list_files:
            df = pd.read_csv(d, sep=';', header=0)
            model = row_name(d)

            df.insert(0, "Model", model)
            df.drop(columns=list(df.filter(regex='Unnamed')), inplace=True)
            df.drop(columns=['File', 'MSE', 'No_missing_lbls'], inplace=True)
            dataframes.append(df)

        full_table = pd.concat(dataframes)
        if args.removemetrics is not None:
            full_table = full_table.drop(columns=args.removemetrics)
        mean_table = full_table.copy()
        # mean_table.insert(column='Type', value='Avg.', loc=1)
        # mean_table = mean_table.groupby(['Type', 'Model']).mean().round(3)
        mean_table = mean_table.groupby(['Model'])
        hd95 = mean_table.HD.quantile(0.95).map('{:.2f}'.format)
        mean_table = mean_table.mean().round(3)

        std_table = full_table.copy()
        # std_table.insert(column='Type', value='STD', loc=1)
        # std_table = std_table.groupby(['Type', 'Model']).std().round(3)
        std_table = std_table.groupby(['Model']).std().round(3)

        # metrics_table = pd.concat([mean_table, std_table]).swaplevel(axis='rows')
        metrics_table = mean_table.applymap('{:.2f}'.format) + u"\u00B1" + std_table.applymap('{:.2f}'.format)
        time_col = metrics_table.pop('Time')
        metrics_table.insert(len(metrics_table.columns), 'Time', time_col)
        metrics_table.insert(5, 'HD 95%ile', hd95)

        metrics_file = os.path.join(args.output, args.filename + '.tex')
        if os.path.exists(metrics_file) and args.overwrite:
            shutil.rmtree(metrics_file, ignore_errors=True)
            metrics_table.to_latex(metrics_file,
                                   column_format='l' + 'c' * len(metrics_table.columns),
                                   caption='Average and standard deviation of the metrics: MSE, NCC, SSIM, DICE and HD. As well as the number of missing labels in the predicted images.')
        elif os.path.exists(metrics_file):
            warnings.warn('File {} already exists. Skipping'.format(metrics_file))
        else:
            metrics_table.to_latex(metrics_file,
                                   column_format='l' + 'c' * len(metrics_table.columns),
                                   caption='Average and standard deviation of the metrics: MSE, NCC, SSIM, DICE and HD. As well as the number of missing labels in the predicted images.')

        print('Done')
    else:
        print('No files found in {}!'.format(args.dir))
