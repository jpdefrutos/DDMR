import pandas as pd
import os
from argparse import ArgumentParser
import re


DICT_MODEL_NAMES = {'BASELINE': 'BL',
                    'SEGGUIDED': 'SG',
                    'UW': 'UW'}

DICT_METRICS_NAMES = {'NCC': 'N',
                      'SSIM': 'S',
                      'DICE': 'D',
                      'DICE MACRO': 'D',
                      'HD': 'H', }


DF_COLS = ['SSIM', 'NCC', 'MSE', 'DICE_MACRO', 'HD', 'HD95', 'Time', 'TRE', 'Experiment', 'Model']


def get_model_name(in_path: str) -> str:
    model = re.search('((UW|SEGGUIDED|BASELINE).*)_\d', in_path)
    if model:
        model = model.group(1).rstrip('_')
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
    else:
        try:
            model = re.search('(SyNCC|SyN)', in_path).group(1)
        except AttributeError:
            raise ValueError('Unknown folder name/model: '+ in_path)
        return model


def find_metric_files(root_path: str, folder_filter: str) -> dict:
    metric_files = dict()
    starting_level = root_path.count(os.sep)
    for r, d, f in os.walk(root_path):
        level = r.count(os.sep) - starting_level
        if level < 3:
            for name in f:
                if 'metrics.csv' == name and folder_filter in r.split(os.sep):
                    model = get_model_name(os.path.join(r, name))
                    metric_files[model] = os.path.join(r, name)
    return metric_files


def read_metrics_files(metrics: dict, experiment: str) -> pd.DataFrame:
    df = pd.DataFrame(columns=DF_COLS)
    for k in metrics.keys():
        csv = pd.read_csv(metrics[k], sep=';')
        csv['Experiment'] = experiment
        csv['Model'] = k
        df = df.append(csv[DF_COLS], ignore_index=True)
    return df


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output-dir', help='Output directory', default='./')
    parser.add_argument('--ixi-runs', help='Directory were the evaluation outputs are stored')
    parser.add_argument('--comet-runs', help='Directory were the evaluation outputs are stored')
    parser.add_argument('--comet-tl-freezenone-runs', help='Directory were the evaluation outputs are stored')
    parser.add_argument('--comet-tl-encoder-runs', help='Directory were the evaluation outputs are stored')
    parser.add_argument('--ants-runs', help='Directory were the evaluation outputs are stored')
    parser.add_argument('--folder-filter', default='Evaluate')

    args = parser.parse_args()

    assert os.path.exists(args.ixi_runs), 'IXI directory not found'
    assert os.path.exists(args.comet_runs), 'COMET directory not found'
    assert os.path.exists(args.comet_tl_freezenone_runs), 'COMET TL Fine Tuned Froze None directory not found'
    assert os.path.exists(args.comet_tl_encoder_runs), 'COMET TL Fine Tuned in 2 Steps directory not found'
    assert os.path.exists(args.ants_runs), 'COMET TL Fine Tuned in 2 Steps directory not found'

    IXI_metrics = find_metric_files(args.ixi_runs, args.folder_filter)
    COMET_metrics = find_metric_files(args.comet_runs, args.folder_filter)
    COMET_TL_FTFN_metrics = find_metric_files(args.comet_tl_freezenone_runs, args.folder_filter)
    COMET_TL_FT2S_metrics = find_metric_files(args.comet_tl_encoder_runs, args.folder_filter)
    ANTS_metrics = find_metric_files(args.ants_runs, args.folder_filter)

    IXI_df = read_metrics_files(IXI_metrics, 'IXI')
    COMET_df = read_metrics_files(COMET_metrics, 'COMET')
    COMET_TL_FtFn_df = read_metrics_files(COMET_TL_FTFN_metrics, 'COMET_TL_FtFn')
    COMET_TL_Ft2S_df = read_metrics_files(COMET_TL_FT2S_metrics, 'COMET_TL_Ft2Stp')
    ANTS_df = read_metrics_files(ANTS_metrics, 'ANTs')

    df = pd.concat([IXI_df, COMET_df, COMET_TL_FtFn_df, COMET_TL_Ft2S_df, ANTS_df], ignore_index=True)
    out_file_path = os.path.join(args.output_dir, 'Combined_metrics.csv')
    if os.path.exists(out_file_path):
        os.remove(out_file_path)
    df.to_csv(out_file_path, sep=';', index=False)
    print('Output file: ' + out_file_path)
