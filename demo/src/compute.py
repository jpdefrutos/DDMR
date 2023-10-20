import subprocess as sp


def run_model(fixed_path, moving_path, output_path):
    sp.check_call(["ddmr", "--fixed", fixed_path, "--moving", moving_path, \
                   "-o", output_path, "-a", "B", "--model", "BL-NS", "--original-resolution"])
