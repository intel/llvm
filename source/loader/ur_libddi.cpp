/*
 *
 * Copyright (C) 2022 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

__urdlllocal ur_result_t context_t::ddiInit() {
    ur_result_t result = UR_RESULT_SUCCESS;

    if (UR_RESULT_SUCCESS == result) {
        result = urGetGlobalProcAddrTable(UR_API_VERSION_CURRENT,
                                          &urDdiTable.Global);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = urGetBindlessImagesExpProcAddrTable(
            UR_API_VERSION_CURRENT, &urDdiTable.BindlessImagesExp);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = urGetCommandBufferExpProcAddrTable(
            UR_API_VERSION_CURRENT, &urDdiTable.CommandBufferExp);
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
        result = urGetEnqueueExpProcAddrTable(UR_API_VERSION_CURRENT,
                                              &urDdiTable.EnqueueExp);
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
        result = urGetKernelExpProcAddrTable(UR_API_VERSION_CURRENT,
                                             &urDdiTable.KernelExp);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = urGetMemProcAddrTable(UR_API_VERSION_CURRENT, &urDdiTable.Mem);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = urGetPhysicalMemProcAddrTable(UR_API_VERSION_CURRENT,
                                               &urDdiTable.PhysicalMem);
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
        result = urGetProgramExpProcAddrTable(UR_API_VERSION_CURRENT,
                                              &urDdiTable.ProgramExp);
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
        result = urGetTensorMapExpProcAddrTable(UR_API_VERSION_CURRENT,
                                                &urDdiTable.TensorMapExp);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = urGetUSMProcAddrTable(UR_API_VERSION_CURRENT, &urDdiTable.USM);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = urGetUSMExpProcAddrTable(UR_API_VERSION_CURRENT,
                                          &urDdiTable.USMExp);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = urGetUsmP2PExpProcAddrTable(UR_API_VERSION_CURRENT,
                                             &urDdiTable.UsmP2PExp);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = urGetVirtualMemProcAddrTable(UR_API_VERSION_CURRENT,
                                              &urDdiTable.VirtualMem);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = urGetDeviceProcAddrTable(UR_API_VERSION_CURRENT,
                                          &urDdiTable.Device);
    }

    return result;
}

} // namespace ur_lib
