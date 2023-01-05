#! /usr/bin/env python3
"""
 Copyright (C) 2022 Intel Corporation

 SPDX-License-Identifier: MIT

"""

import pytest
import sys
import os

import include.ur as ur

def test_ddi():
        ddi = ur.UR_DDI(ur.ur_api_version_v.CURRENT)
        assert True
