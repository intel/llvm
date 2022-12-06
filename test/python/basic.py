#! /usr/bin/env python3
"""
 Copyright (C) 2022 Intel Corporation

 SPDX-License-Identifier: MIT

"""

import pytest
import sys
import os

# not ideal, but this solution is simple and portable. Alternative is to set
# PYTHONPATH or test on installed sources.
sys.path.insert(1, '../include')
import ur

def test_ddi():
        ddi = ur.UR_DDI(ur.ur_api_version_v.CURRENT);
        assert True
