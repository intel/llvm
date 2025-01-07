# See check-correctness-of-requirements.cpp

import os
import sys

input_data = sys.argv[1]
current_dir = os.path.dirname(os.path.abspath(__file__))

# See sycl/test-e2e/sycl_lit_features.py.
sys.path.append(current_dir + "/../../test-e2e")
from sycl_lit_features import get_all_sycl_lit_features

all_features = get_all_sycl_lit_features()

exit_code = 0
with open(input_data, "r") as file:
    requirements = set(file.read().split())
    for requirement in requirements:
        if not requirement in all_features:
            exit_code = 1
            print("Unsupported requirement: " + requirement)

sys.exit(exit_code)
