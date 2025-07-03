/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file sanitizer_utils.cpp
 *
 */

#include "sanitizer_utils.hpp"
#include "sanitizer_common/sanitizer_common.hpp"

namespace ur_sanitizer_layer {

namespace {

ur_usm_type_t GetUSMType(ur_context_handle_t Context, const void *MemPtr) {
  ur_usm_type_t USMType = UR_USM_TYPE_UNKNOWN;
  [[maybe_unused]] auto Result =
      getContext()->urDdiTable.USM.pfnGetMemAllocInfo(
          Context, MemPtr, UR_USM_ALLOC_INFO_TYPE, sizeof(USMType), &USMType,
          nullptr);
  assert(Result == UR_RESULT_SUCCESS);
  return USMType;
}

} // namespace

ManagedQueue::ManagedQueue(ur_context_handle_t Context,
                           ur_device_handle_t Device) {
  [[maybe_unused]] auto Result = getContext()->urDdiTable.Queue.pfnCreate(
      Context, Device, nullptr, &Handle);
  assert(Result == UR_RESULT_SUCCESS && "Failed to create ManagedQueue");
  UR_LOG_L(getContext()->logger, DEBUG, ">>> ManagedQueue {}", (void *)Handle);
}

ManagedQueue::~ManagedQueue() {
  UR_LOG_L(getContext()->logger, DEBUG, "<<< ~ManagedQueue {}", (void *)Handle);

  [[maybe_unused]] ur_result_t Result;
  Result = getContext()->urDdiTable.Queue.pfnFinish(Handle);
  if (Result != UR_RESULT_SUCCESS) {
    UR_LOG_L(getContext()->logger, ERR, "Failed to finish ManagedQueue: {}",
             Result);
  }
  assert(Result == UR_RESULT_SUCCESS && "Failed to finish ManagedQueue");
  Result = getContext()->urDdiTable.Queue.pfnRelease(Handle);
  assert(Result == UR_RESULT_SUCCESS && "Failed to release ManagedQueue");
}

ur_context_handle_t GetContext(ur_queue_handle_t Queue) {
  ur_context_handle_t Context{};
  [[maybe_unused]] auto Result = getContext()->urDdiTable.Queue.pfnGetInfo(
      Queue, UR_QUEUE_INFO_CONTEXT, sizeof(ur_context_handle_t), &Context,
      nullptr);
  assert(Result == UR_RESULT_SUCCESS && "getContext() failed");
  return Context;
}

ur_context_handle_t GetContext(ur_program_handle_t Program) {
  ur_context_handle_t Context{};
  [[maybe_unused]] auto Result = getContext()->urDdiTable.Program.pfnGetInfo(
      Program, UR_PROGRAM_INFO_CONTEXT, sizeof(ur_context_handle_t), &Context,
      nullptr);
  assert(Result == UR_RESULT_SUCCESS && "getContext() failed");
  return Context;
}

ur_context_handle_t GetContext(ur_kernel_handle_t Kernel) {
  ur_context_handle_t Context{};
  [[maybe_unused]] auto Result = getContext()->urDdiTable.Kernel.pfnGetInfo(
      Kernel, UR_KERNEL_INFO_CONTEXT, sizeof(ur_context_handle_t), &Context,
      nullptr);
  assert(Result == UR_RESULT_SUCCESS && "getContext() failed");
  return Context;
}

ur_device_handle_t GetDevice(ur_queue_handle_t Queue) {
  ur_device_handle_t Device{};
  [[maybe_unused]] auto Result = getContext()->urDdiTable.Queue.pfnGetInfo(
      Queue, UR_QUEUE_INFO_DEVICE, sizeof(ur_device_handle_t), &Device,
      nullptr);
  assert(Result == UR_RESULT_SUCCESS && "getDevice() failed");
  return Device;
}

std::vector<ur_device_handle_t> GetDevices(ur_context_handle_t Context) {
  std::vector<ur_device_handle_t> Devices{};
  uint32_t DeviceNum = 0;
  [[maybe_unused]] ur_result_t Result;
  Result = getContext()->urDdiTable.Context.pfnGetInfo(
      Context, UR_CONTEXT_INFO_NUM_DEVICES, sizeof(uint32_t), &DeviceNum,
      nullptr);
  assert(Result == UR_RESULT_SUCCESS && "getDevices(Context) failed");
  Devices.resize(DeviceNum);
  Result = getContext()->urDdiTable.Context.pfnGetInfo(
      Context, UR_CONTEXT_INFO_DEVICES, sizeof(ur_device_handle_t) * DeviceNum,
      Devices.data(), nullptr);
  assert(Result == UR_RESULT_SUCCESS && "getDevices(Context) failed");
  return Devices;
}

ur_program_handle_t GetProgram(ur_kernel_handle_t Kernel) {
  ur_program_handle_t Program{};
  [[maybe_unused]] auto Result = getContext()->urDdiTable.Kernel.pfnGetInfo(
      Kernel, UR_KERNEL_INFO_PROGRAM, sizeof(ur_program_handle_t), &Program,
      nullptr);
  assert(Result == UR_RESULT_SUCCESS && "getProgram() failed");
  return Program;
}

size_t GetDeviceLocalMemorySize(ur_device_handle_t Device) {
  size_t LocalMemorySize{};
  [[maybe_unused]] auto Result = getContext()->urDdiTable.Device.pfnGetInfo(
      Device, UR_DEVICE_INFO_LOCAL_MEM_SIZE, sizeof(LocalMemorySize),
      &LocalMemorySize, nullptr);
  assert(Result == UR_RESULT_SUCCESS && "getLocalMemorySize() failed");
  return LocalMemorySize;
}

std::string GetKernelName(ur_kernel_handle_t Kernel) {
  size_t KernelNameSize = 0;
  [[maybe_unused]] auto Result = getContext()->urDdiTable.Kernel.pfnGetInfo(
      Kernel, UR_KERNEL_INFO_FUNCTION_NAME, 0, nullptr, &KernelNameSize);
  assert(Result == UR_RESULT_SUCCESS && "getKernelName() failed");

  std::vector<char> KernelNameBuf(KernelNameSize);
  Result = getContext()->urDdiTable.Kernel.pfnGetInfo(
      Kernel, UR_KERNEL_INFO_FUNCTION_NAME, KernelNameSize,
      KernelNameBuf.data(), nullptr);
  assert(Result == UR_RESULT_SUCCESS && "getKernelName() failed");

  return std::string(KernelNameBuf.data(), KernelNameSize - 1);
}

bool IsUSM(ur_context_handle_t Context, const void *MemPtr) {
  ur_usm_type_t USMType = GetUSMType(Context, MemPtr);
  return USMType != UR_USM_TYPE_UNKNOWN;
}

bool IsHostUSM(ur_context_handle_t Context, const void *MemPtr) {
  ur_usm_type_t USMType = GetUSMType(Context, MemPtr);
  return USMType == UR_USM_TYPE_HOST;
}

ur_device_handle_t GetUSMAllocDevice(ur_context_handle_t Context,
                                     const void *MemPtr) {
  assert(IsUSM(Context, MemPtr));
  ur_device_handle_t Device{};
  getContext()->urDdiTable.USM.pfnGetMemAllocInfo(
      Context, MemPtr, UR_USM_ALLOC_INFO_DEVICE, sizeof(Device), &Device,
      nullptr);
  return Device;
}

ur_device_handle_t GetUSMAllocDevice(ur_queue_handle_t Queue,
                                     const void *MemPtr) {
  ur_context_handle_t Context = GetContext(Queue);
  assert(Context && IsUSM(Context, MemPtr));
  return IsHostUSM(Context, MemPtr) ? GetDevice(Queue)
                                    : GetUSMAllocDevice(Context, MemPtr);
}

DeviceType GetDeviceType(ur_context_handle_t Context,
                         ur_device_handle_t Device) {
  ur_device_type_t DeviceType = UR_DEVICE_TYPE_DEFAULT;
  [[maybe_unused]] auto Result = getContext()->urDdiTable.Device.pfnGetInfo(
      Device, UR_DEVICE_INFO_TYPE, sizeof(DeviceType), &DeviceType, nullptr);
  assert(Result == UR_RESULT_SUCCESS && "getDeviceType() failed");
  switch (DeviceType) {
  case UR_DEVICE_TYPE_CPU:
  case UR_DEVICE_TYPE_FPGA:
    // TODO: Check fpga is fpga emulator
    return DeviceType::CPU;
  case UR_DEVICE_TYPE_GPU: {
    uptr Ptr;
    [[maybe_unused]] ur_result_t Result =
        getContext()->urDdiTable.USM.pfnDeviceAlloc(Context, Device, nullptr,
                                                    nullptr, 4, (void **)&Ptr);
    UR_LOG_L(getContext()->logger, DEBUG, "GetDeviceType: {}", (void *)Ptr);
    assert(Result == UR_RESULT_SUCCESS &&
           "getDeviceType() failed at allocating device USM");
    // FIXME: There's no API querying the address bits of device, so we guess it
    // by the value of device USM pointer (see "USM Allocation Range" in
    // asan_shadow.cpp)
    auto Type = DeviceType::UNKNOWN;

    // L0 changes their VA layout.
    // TODO: update our shadow memory layout/algorithms to accordingly.
    if (Ptr >> 52 == 0xff0U) {
      Type = DeviceType::GPU_PVC;
    } else {
      Type = DeviceType::GPU_DG2;
    }
    Result = getContext()->urDdiTable.USM.pfnFree(Context, (void *)Ptr);
    assert(Result == UR_RESULT_SUCCESS &&
           "getDeviceType() failed at releasing device USM");
    return Type;
  }
  default:
    return DeviceType::UNKNOWN;
  }
}

ur_device_handle_t GetParentDevice(ur_device_handle_t Device) {
  ur_device_handle_t ParentDevice{};
  [[maybe_unused]] auto Result = getContext()->urDdiTable.Device.pfnGetInfo(
      Device, UR_DEVICE_INFO_PARENT_DEVICE, sizeof(ur_device_handle_t),
      &ParentDevice, nullptr);
  assert(Result == UR_RESULT_SUCCESS && "getParentDevice() failed");
  return ParentDevice;
}

bool GetDeviceUSMCapability(ur_device_handle_t Device,
                            ur_device_info_t USMInfo) {
  ur_device_usm_access_capability_flags_t Flag;
  [[maybe_unused]] auto Result = getContext()->urDdiTable.Device.pfnGetInfo(
      Device, USMInfo, sizeof(Flag), &Flag, nullptr);
  return (bool)Flag;
}

std::vector<ur_device_handle_t> GetDevices(ur_program_handle_t Program) {
  uint32_t DeviceNum = 0;
  [[maybe_unused]] ur_result_t Result =
      getContext()->urDdiTable.Program.pfnGetInfo(
          Program, UR_PROGRAM_INFO_NUM_DEVICES, sizeof(DeviceNum), &DeviceNum,
          nullptr);
  assert(Result == UR_RESULT_SUCCESS && "getDevices(Program) failed");

  std::vector<ur_device_handle_t> Devices;
  Devices.resize(DeviceNum);
  Result = getContext()->urDdiTable.Program.pfnGetInfo(
      Program, UR_PROGRAM_INFO_DEVICES, DeviceNum * sizeof(ur_device_handle_t),
      Devices.data(), nullptr);
  assert(Result == UR_RESULT_SUCCESS && "getDevices(Program) failed");

  return Devices;
}

uint32_t GetKernelNumArgs(ur_kernel_handle_t Kernel) {
  uint32_t NumArgs = 0;
  [[maybe_unused]] auto Res = getContext()->urDdiTable.Kernel.pfnGetInfo(
      Kernel, UR_KERNEL_INFO_NUM_ARGS, sizeof(NumArgs), &NumArgs, nullptr);
  assert(Res == UR_RESULT_SUCCESS);
  return NumArgs;
}

size_t GetKernelLocalMemorySize(ur_kernel_handle_t Kernel,
                                ur_device_handle_t Device) {
  size_t Size = 0;
  [[maybe_unused]] auto Res = getContext()->urDdiTable.Kernel.pfnGetGroupInfo(
      Kernel, Device, UR_KERNEL_GROUP_INFO_LOCAL_MEM_SIZE, sizeof(size_t),
      &Size, nullptr);
  assert(Res == UR_RESULT_SUCCESS);
  return Size;
}

size_t GetKernelPrivateMemorySize(ur_kernel_handle_t Kernel,
                                  ur_device_handle_t Device) {
  size_t Size = 0;
  [[maybe_unused]] auto Res = getContext()->urDdiTable.Kernel.pfnGetGroupInfo(
      Kernel, Device, UR_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE, sizeof(size_t),
      &Size, nullptr);
  assert(Res == UR_RESULT_SUCCESS);
  return Size;
}

size_t GetVirtualMemGranularity(ur_context_handle_t Context,
                                ur_device_handle_t Device) {
  size_t Size;
  [[maybe_unused]] auto Result =
      getContext()->urDdiTable.VirtualMem.pfnGranularityGetInfo(
          Context, Device, UR_VIRTUAL_MEM_GRANULARITY_INFO_RECOMMENDED,
          sizeof(Size), &Size, nullptr);
  assert(Result == UR_RESULT_SUCCESS);
  return Size;
}

void PrintUrBuildLogIfError(ur_result_t Result, ur_program_handle_t Program,
                            ur_device_handle_t *Devices, size_t NumDevices) {
  if (Result == UR_RESULT_SUCCESS ||
      Result == UR_RESULT_ERROR_UNSUPPORTED_FEATURE)
    return;

  if (!Program || !Devices || NumDevices == 0) {
    UR_LOG_L(getContext()->logger, ERR, "Failed to get build log.");
    return;
  }

  UR_LOG_L(getContext()->logger, ERR, "Printing build log for program {}",
           (void *)Program);
  for (size_t I = 0; I < NumDevices; I++) {
    std::vector<char> LogBuf;
    size_t LogSize = 0;
    auto Device = Devices[I];

    auto UrRes = getContext()->urDdiTable.Program.pfnGetBuildInfo(
        Program, Device, UR_PROGRAM_BUILD_INFO_LOG, 0, nullptr, &LogSize);
    if (UrRes != UR_RESULT_SUCCESS) {
      UR_LOG_L(getContext()->logger, ERR,
               "For device {}: failed to get build log size.", (void *)Device);
      continue;
    }

    LogBuf.resize(LogSize);
    UrRes = getContext()->urDdiTable.Program.pfnGetBuildInfo(
        Program, Device, UR_PROGRAM_BUILD_INFO_LOG, LogSize, LogBuf.data(),
        nullptr);
    if (UrRes != UR_RESULT_SUCCESS) {
      UR_LOG_L(getContext()->logger, ERR,
               "For device {}: failed to get build log.", (void *)Device);
      continue;
    }

    UR_LOG_L(getContext()->logger, ERR, "For device {}:\n{}", (void *)Device,
             LogBuf.data());
  }
}

} // namespace ur_sanitizer_layer
