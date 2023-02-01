import numpy as np
import statsmodels
import scipy
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import fdrcorrection
from scipy import stats
import pandas as pd

import argparse
import os

# increase length of string in pandas
pd.options.display.max_colwidth = 100


def post_hoc_ixi(df_in: pd.DataFrame, outdir: str = './'):
    df = df_in[df_in["Experiment"] == "IXI"]

    TRE_values = df["TRE"]
    m_comp = pairwise_tukeyhsd(df["TRE"], df["Model"], alpha=0.05)
    model_names = np.unique(df["Model"])

    all_pvalues = -1 * np.ones((len(model_names), len(model_names)), dtype=np.float32)
    pvs = m_comp.pvalues
    cnt = 0
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            all_pvalues[i, j] = pvs[cnt]
            cnt += 1
    all_pvalues = np.round(all_pvalues, 6)
    all_pvalues = all_pvalues[:-1, 1:]

    col_new_names = ["\textbf{\rot{\multicolumn{1}{r}{" + n + "}}}" for n in model_names]

    out_pd = pd.DataFrame(data=all_pvalues, index=model_names[:-1], columns=col_new_names[1:])
    stack = out_pd.stack()
    stack[(0 < stack) & (stack <= 0.001)] = '\cellcolor{green!25}$<$0.001'

    for i in range(stack.shape[0]):
        try:
            curr = stack[i]
            if (float(curr) > 0.0011) & (float(curr) < 0.05):
                stack[i] = '\cellcolor{green!50}' + str(np.round(stack[i], 3))
            elif (float(curr) >= 0.05) & (float(curr) < 0.1):
                stack[i] = '\cellcolor{red!50}' + str(np.round(stack[i], 3))
            elif (float(curr) >= 0.1):
                stack[i] = '\cellcolor{red!25}' + str(np.round(stack[i], 3))
        except Exception:
            continue

    out_pd = stack.unstack()
    out_pd = out_pd.replace(-1.0, "-")
    out_pd = out_pd.replace(-0.0, '\cellcolor{green!25}$<$0.001')

    with open(os.path.join(outdir, "tukey_pvalues_result_IXI.txt"), "w") as pfile:
        pfile.write("{}".format(out_pd.to_latex(escape=False, column_format="r" + "c"*all_pvalues.shape[1], bold_rows=True)))
    
    print(out_pd)


def study_seg_guided_ANTs(df_in: pd.DataFrame):
    tre_sg = df_in[(df_in["Experiment"] == "COMET_TL_Ft2Stp") & (df_in["Model"] == "SG-NSD")]["TRE"]
    tre_syn = df_in[(df_in["Experiment"] == "ANTs") & (df_in["Model"] == "SyN")]["TRE"]
    tre_syncc = df_in[(df_in["Experiment"] == "ANTs") & (df_in["Model"] == "SyNCC")]["TRE"]

    pvs = list()
    pvs.append(stats.wilcoxon(tre_sg, tre_syn, alternative="less").pvalue)
    pvs.append(stats.wilcoxon(tre_sg, tre_syncc, alternative="less").pvalue)

    # False discovery rate to get corrected p-values
    corrected_pvs = fdrcorrection(pvs, alpha=0.05, method="indep")[1]  # Benjamini/Hochberg -> method="indep"

    print("SG-NSD vs SyN:", corrected_pvs[0])
    print("SG-NSD vs SyNCC:", corrected_pvs[1])


def study_baseline_ANTs(df_in: pd.DataFrame):
    tre_bl = df_in[(df_in["Experiment"] == "COMET_TL_Ft2Stp") & (df_in["Model"] == "BL-N")]["TRE"]
    tre_syn = df_in[(df_in["Experiment"] == "ANTs") & (df_in["Model"] == "SyN")]["TRE"]
    tre_syncc = df_in[(df_in["Experiment"] == "ANTs") & (df_in["Model"] == "SyNCC")]["TRE"]

    pvs = list()
    pvs.append(stats.wilcoxon(tre_bl, tre_syn, alternative="less").pvalue)
    pvs.append(stats.wilcoxon(tre_bl, tre_syncc, alternative="less").pvalue)

    # False discovery rate to get corrected p-values
    corrected_pvs = fdrcorrection(pvs, alpha=0.05, method="indep")[1]  # Benjamini/Hochberg -> method="indep"

    print("BL-N vs SyN:", corrected_pvs[0])
    print("BL-N vs SyNCC:", corrected_pvs[1])


def study_transfer_learning_benefit(df_in: pd.DataFrame):
    df_tl = df_in[df_in["Experiment"] == "COMET_TL_Ft2Stp"]
    df_orig = df[df["Experiment"] == "COMET"]

    pvs = []
    for model in ["BL-N", "SG-NSD", "UW-NSD"]:

        curr_tl = df_tl[df_tl["Model"] == model]
        curr_orig = df_orig[df_orig["Model"] == model]

        TRE_tl = curr_tl["TRE"]
        TRE_orig = curr_orig["TRE"]

        # perform non-parametric hypothesis test to assess significance
        ret = stats.wilcoxon(TRE_tl, TRE_orig, alternative="less")
        pv = ret.pvalue
        pvs.append(pv)

    # False discovery rate to get corrected p-values
    corrected_pvs = fdrcorrection(pvs, alpha=0.05, method="indep")[1]  # Benjamini/Hochberg -> method="indep"

    print("BL-N:", corrected_pvs[0])
    print("SG-NSD:", corrected_pvs[1])
    print("UW-NSD:", corrected_pvs[2])

def post_hoc_comet(df_in: pd.DataFrame):
    df_tl = df_in[df_in["Experiment"] == "COMET_TL_Ft2Stp"]

    filter_ = np.array([x in ["BL-N", "SG-NSD", "UW-NSD"] for x in df_tl["Model"]])

    df_tl = df_tl[filter_]

    # Is TRE in SG-NSD significantly lower than TRE in BL-N?
    ret1 = stats.wilcoxon(
        df_tl[df_tl["Model"] == "SG-NSD"]["TRE"],
        df_tl[df_tl["Model"] == "BL-N"]["TRE"],
        alternative="less"
    )
    pv1 = ret1.pvalue

    # Is TRE in UW-NSD significantly lower than TRE in SG_NSD?
    ret2 = stats.wilcoxon(
        df_tl[df_tl["Model"] == "UW-NSD"]["TRE"],
        df_tl[df_tl["Model"] == "SG-NSD"]["TRE"],
        alternative="less"
    )
    pv2 = ret2.pvalue

    # False discovery rate to get corrected p-values
    pvs = [pv1, pv2]
    corrected_pvs = fdrcorrection(pvs, alpha=0.05, method="indep")[1]  # Benjamini/Hochberg -> method="indep"

    print("Seg-guiding benefit:", corrected_pvs[0])
    print("Uncertainty-weighting benefit:", corrected_pvs[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', type=str, help='CSV file with the metrics')
    parser.add_argument('--sep', type=str, help='CSV Separator (default: ;)', default=';')
    parser.add_argument('--outdir', type=str, help='Output directory (default: .)', default='./')
    args = parser.parse_args()

    assert os.path.exists(args.file), 'CSV file not found'
    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.file, sep=args.sep)
    if "Unnamed" in df.columns[0]:
        df = df.iloc[:, 1:]
    if any(['_' in m for m in df["Model"].unique()]):
        df["Model"] = [x.replace("_", "-") for x in df["Model"]]

    print("\nComparing all contrasts in TRE of all models in the IXI dataset:")
    post_hoc_ixi(df, args.outdir)

    print("\nTransfer learning benefit (COMET):")
    study_transfer_learning_benefit(df)

    print("\nAssessing whether there is a benefit to segmentation-guiding and uncertainty weighting (COMET):")
    post_hoc_comet(df)

    print("\nAssessing the performance of BL-N vs ANTs registration (COMET):")
    study_baseline_ANTs(df)

    print("\nAssessing the performance of SG-NSD vs ANTs registration (COMET):")
    study_seg_guided_ANTs(df)

    print('Statsmodels v.: ' + statsmodels.__version__)
    print('Scipy v.: ' + scipy.__version__)
