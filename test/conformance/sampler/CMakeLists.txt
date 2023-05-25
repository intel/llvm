# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

add_conformance_test_with_devices_environment(sampler 
    urSamplerCreate.cpp
    urSamplerCreateWithNativeHandle.cpp 
    urSamplerGetInfo.cpp 
    urSamplerGetNativeHandle.cpp 
    urSamplerRelease.cpp 
    urSamplerRetain.cpp
)
