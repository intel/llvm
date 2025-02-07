# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_conformance_test_with_kernels_environment(usm
    urUSMDeviceAlloc.cpp
    urUSMFree.cpp
    urUSMGetMemAllocInfo.cpp
    urUSMHostAlloc.cpp
    urUSMPoolCreate.cpp
    urUSMPoolGetInfo.cpp
    urUSMPoolRelease.cpp
    urUSMPoolRetain.cpp
    urUSMSharedAlloc.cpp)

if(UMF_BUILD_SHARED_LIBRARY)
    add_conformance_test_with_kernels_environment(usm_umf
        umfIntegration.cpp)

    target_link_libraries(test-usm_umf PRIVATE umf::umf)
else()
    message(WARNING "UMF build as static library, UMF integration tests will be disabled")
endif()
