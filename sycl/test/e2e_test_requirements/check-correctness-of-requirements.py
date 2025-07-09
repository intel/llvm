# See check-correctness-of-requirements.cpp

import os
import sys

input_data = sys.argv[1]
current_dir = os.path.dirname(os.path.abspath(__file__))

# See sycl/test-e2e/sycl_lit_allowed_features.py.
sys.path.append(current_dir + "/../../test-e2e")
from sycl_lit_allowed_features import get_sycl_lit_allowed_features

allowed_features = get_sycl_lit_allowed_features()

exit_code = 0
with open(input_data, "r") as file:
    requirements = set(file.read().split())
    for requirement in requirements:
        if not requirement in allowed_features:
            exit_code = 1
            print("Unsupported requirement: " + requirement)

sys.exit(exit_code)
