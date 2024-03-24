/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ur_sanitizer_utils.cpp
 *
 */

#include "ur_sanitizer_utils.hpp"
#include "ur_sanitizer_layer.hpp"

namespace ur_sanitizer_layer {

ManagedQueue::ManagedQueue(ur_context_handle_t Context,
                           ur_device_handle_t Device) {
    [[maybe_unused]] auto Result =
        context.urDdiTable.Queue.pfnCreate(Context, Device, nullptr, &Handle);
    assert(Result == UR_RESULT_SUCCESS);
    context.logger.debug(">>> ManagedQueue {}", (void *)Handle);
}

ManagedQueue::~ManagedQueue() {
    context.logger.debug("<<< ~ManagedQueue {}", (void *)Handle);

    [[maybe_unused]] ur_result_t Result;
    Result = context.urDdiTable.Queue.pfnFinish(Handle);
    if (Result != UR_RESULT_SUCCESS) {
        context.logger.error("Failed to finish ManagedQueue: {}", Result);
    }
    assert(Result == UR_RESULT_SUCCESS);
    Result = context.urDdiTable.Queue.pfnRelease(Handle);
    assert(Result == UR_RESULT_SUCCESS);
}

ur_context_handle_t GetContext(ur_queue_handle_t Queue) {
    ur_context_handle_t Context{};
    [[maybe_unused]] auto Result = context.urDdiTable.Queue.pfnGetInfo(
        Queue, UR_QUEUE_INFO_CONTEXT, sizeof(ur_context_handle_t), &Context,
        nullptr);
    assert(Result == UR_RESULT_SUCCESS && "getContext() failed");
    return Context;
}

ur_context_handle_t GetContext(ur_program_handle_t Program) {
    ur_context_handle_t Context{};
    [[maybe_unused]] auto Result = context.urDdiTable.Program.pfnGetInfo(
        Program, UR_PROGRAM_INFO_CONTEXT, sizeof(ur_context_handle_t), &Context,
        nullptr);
    assert(Result == UR_RESULT_SUCCESS && "getContext() failed");
    return Context;
}

ur_context_handle_t GetContext(ur_kernel_handle_t Kernel) {
    ur_context_handle_t Context{};
    [[maybe_unused]] auto Result = context.urDdiTable.Kernel.pfnGetInfo(
        Kernel, UR_KERNEL_INFO_CONTEXT, sizeof(ur_context_handle_t), &Context,
        nullptr);
    assert(Result == UR_RESULT_SUCCESS && "getContext() failed");
    return Context;
}

ur_device_handle_t GetDevice(ur_queue_handle_t Queue) {
    ur_device_handle_t Device{};
    [[maybe_unused]] auto Result = context.urDdiTable.Queue.pfnGetInfo(
        Queue, UR_QUEUE_INFO_DEVICE, sizeof(ur_device_handle_t), &Device,
        nullptr);
    assert(Result == UR_RESULT_SUCCESS && "getDevice() failed");
    return Device;
}

ur_program_handle_t GetProgram(ur_kernel_handle_t Kernel) {
    ur_program_handle_t Program{};
    [[maybe_unused]] auto Result = context.urDdiTable.Kernel.pfnGetInfo(
        Kernel, UR_KERNEL_INFO_PROGRAM, sizeof(ur_program_handle_t), &Program,
        nullptr);
    assert(Result == UR_RESULT_SUCCESS && "getProgram() failed");
    return Program;
}

size_t GetLocalMemorySize(ur_device_handle_t Device) {
    size_t LocalMemorySize{};
    [[maybe_unused]] auto Result = context.urDdiTable.Device.pfnGetInfo(
        Device, UR_DEVICE_INFO_LOCAL_MEM_SIZE, sizeof(LocalMemorySize),
        &LocalMemorySize, nullptr);
    assert(Result == UR_RESULT_SUCCESS && "getLocalMemorySize() failed");
    return LocalMemorySize;
}

std::string GetKernelName(ur_kernel_handle_t Kernel) {
    size_t KernelNameSize = 0;
    [[maybe_unused]] auto Result = context.urDdiTable.Kernel.pfnGetInfo(
        Kernel, UR_KERNEL_INFO_FUNCTION_NAME, 0, nullptr, &KernelNameSize);
    assert(Result == UR_RESULT_SUCCESS && "getKernelName() failed");

    std::vector<char> KernelNameBuf(KernelNameSize);
    Result = context.urDdiTable.Kernel.pfnGetInfo(
        Kernel, UR_KERNEL_INFO_FUNCTION_NAME, KernelNameSize,
        KernelNameBuf.data(), nullptr);
    assert(Result == UR_RESULT_SUCCESS && "getKernelName() failed");

    return std::string(KernelNameBuf.data(), KernelNameSize - 1);
}

ur_device_handle_t GetUSMAllocDevice(ur_context_handle_t Context,
                                     const void *MemPtr) {
    ur_device_handle_t Device{};
    // if urGetMemAllocInfo failed, return nullptr
    context.urDdiTable.USM.pfnGetMemAllocInfo(Context, MemPtr,
                                              UR_USM_ALLOC_INFO_DEVICE,
                                              sizeof(Device), &Device, nullptr);
    return Device;
}

DeviceType GetDeviceType(ur_device_handle_t Device) {
    ur_device_type_t DeviceType = UR_DEVICE_TYPE_DEFAULT;
    [[maybe_unused]] auto Result = context.urDdiTable.Device.pfnGetInfo(
        Device, UR_DEVICE_INFO_TYPE, sizeof(DeviceType), &DeviceType, nullptr);
    assert(Result == UR_RESULT_SUCCESS && "getDeviceType() failed");
    switch (DeviceType) {
    case UR_DEVICE_TYPE_CPU:
    case UR_DEVICE_TYPE_FPGA:
        // TODO: Check fpga is fpga emulator
        return DeviceType::CPU;
    case UR_DEVICE_TYPE_GPU: {
        // TODO: Check device name
        return DeviceType::GPU_PVC;
    }
    default:
        return DeviceType::UNKNOWN;
    }
}

std::vector<ur_device_handle_t> GetProgramDevices(ur_program_handle_t Program) {
    size_t PropSize;
    [[maybe_unused]] ur_result_t Result = context.urDdiTable.Program.pfnGetInfo(
        Program, UR_PROGRAM_INFO_DEVICES, 0, nullptr, &PropSize);
    assert(Result == UR_RESULT_SUCCESS);

    std::vector<ur_device_handle_t> Devices;
    Devices.resize(PropSize / sizeof(ur_device_handle_t));
    Result = context.urDdiTable.Program.pfnGetInfo(
        Program, UR_PROGRAM_INFO_DEVICES, PropSize, Devices.data(), nullptr);
    assert(Result == UR_RESULT_SUCCESS);

    return Devices;
}

size_t GetKernelNumArgs(ur_kernel_handle_t Kernel) {
    size_t NumArgs = 0;
    [[maybe_unused]] auto Res = context.urDdiTable.Kernel.pfnGetInfo(
        Kernel, UR_KERNEL_INFO_NUM_ARGS, sizeof(NumArgs), &NumArgs, nullptr);
    assert(Res == UR_RESULT_SUCCESS);
    return NumArgs;
}

size_t GetVirtualMemGranularity(ur_context_handle_t Context,
                                ur_device_handle_t Device) {
    size_t Size;
    [[maybe_unused]] auto Result =
        context.urDdiTable.VirtualMem.pfnGranularityGetInfo(
            Context, Device, UR_VIRTUAL_MEM_GRANULARITY_INFO_RECOMMENDED,
            sizeof(Size), &Size, nullptr);
    assert(Result == UR_RESULT_SUCCESS);
    return Size;
}

} // namespace ur_sanitizer_layer
