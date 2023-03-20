/*
 *
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 * @file ur_libddi.cpp
 *
 */
#include "ur_lib.hpp"
#ifndef DYNAMIC_LOAD_LOADER
#include "ur_ddi.h"
#endif

namespace ur_lib {
///////////////////////////////////////////////////////////////////////////////

__urdlllocal ur_result_t context_t::urInit() {
    ur_result_t result = UR_RESULT_SUCCESS;

    if (UR_RESULT_SUCCESS == result) {
        result = urGetGlobalProcAddrTable(UR_API_VERSION_CURRENT,
                                          &urDdiTable.Global);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = urGetContextProcAddrTable(UR_API_VERSION_CURRENT,
                                           &urDdiTable.Context);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = urGetEnqueueProcAddrTable(UR_API_VERSION_CURRENT,
                                           &urDdiTable.Enqueue);
    }

    if (UR_RESULT_SUCCESS == result) {
        result =
            urGetEventProcAddrTable(UR_API_VERSION_CURRENT, &urDdiTable.Event);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = urGetKernelProcAddrTable(UR_API_VERSION_CURRENT,
                                          &urDdiTable.Kernel);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = urGetMemProcAddrTable(UR_API_VERSION_CURRENT, &urDdiTable.Mem);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = urGetPlatformProcAddrTable(UR_API_VERSION_CURRENT,
                                            &urDdiTable.Platform);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = urGetProgramProcAddrTable(UR_API_VERSION_CURRENT,
                                           &urDdiTable.Program);
    }

    if (UR_RESULT_SUCCESS == result) {
        result =
            urGetQueueProcAddrTable(UR_API_VERSION_CURRENT, &urDdiTable.Queue);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = urGetSamplerProcAddrTable(UR_API_VERSION_CURRENT,
                                           &urDdiTable.Sampler);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = urGetUSMProcAddrTable(UR_API_VERSION_CURRENT, &urDdiTable.USM);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = urGetDeviceProcAddrTable(UR_API_VERSION_CURRENT,
                                          &urDdiTable.Device);
    }

    return result;
}

} // namespace ur_lib
