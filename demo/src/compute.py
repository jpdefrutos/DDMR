import subprocess as sp


def run_model(fixed_path=None, moving_path=None, fixed_seg_path=None, moving_seg_path=None, output_path=None, task="B"):
    if (fixed_seg_path is None) or (moving_seg_path is None):
        print("The fixed or moving segmentation were not provided and are thus ignored for inference.")
        sp.check_call(["ddmr", "-f", fixed_path, "-m", moving_path, \
                    "-o", output_path, "-a", task, "--model", "BL-NS", "--original-resolution"])
    else:
        sp.check_call(["ddmr", "-f", fixed_path, "-m", moving_path, "-fs", fixed_seg_path, "-ms", moving_seg_path,
                    "-o", output_path, "-a", task, "--model", "BL-NS", "--original-resolution"])
