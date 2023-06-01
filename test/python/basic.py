#! /usr/bin/env python3
"""
 Copyright (C) 2022 Intel Corporation

 Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 See LICENSE.TXT
 SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""

import pytest
import sys
import os

import include.ur as ur

def test_ddi():
        ddi = ur.UR_DDI(ur.ur_api_version_v.CURRENT)
        assert True
