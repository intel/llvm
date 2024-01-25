//===--------- kernel.hpp - Level Zero Adapter ----------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "common.hpp"
#include "memory.hpp"
#include <unordered_set>

struct ur_kernel_handle_t_ : _ur_object {
  ur_kernel_handle_t_(bool OwnZeHandle, ur_program_handle_t Program)
      : Context{nullptr}, Program{Program}, SubmissionsCount{0}, MemAllocs{} {
    OwnNativeHandle = OwnZeHandle;
  }

  ur_kernel_handle_t_(ze_kernel_handle_t Kernel, bool OwnZeHandle,
                      ur_context_handle_t Context)
      : Context{Context}, Program{nullptr}, ZeKernel{Kernel},
        SubmissionsCount{0}, MemAllocs{} {
    OwnNativeHandle = OwnZeHandle;
  }

  // Keep the program of the kernel.
  ur_context_handle_t Context;

  // Keep the program of the kernel.
  ur_program_handle_t Program;

  // Level Zero function handle.
  ze_kernel_handle_t ZeKernel;

  // Map of L0 kernels created for all the devices for which a UR Program
  // has been built. It may contain duplicated kernel entries for a root
  // device and its sub-devices.
  std::unordered_map<ze_device_handle_t, ze_kernel_handle_t> ZeKernelMap;

  // Vector of L0 kernels. Each entry is unique, so this is used for
  // destroying the kernels instead of ZeKernelMap
  std::vector<ze_kernel_handle_t> ZeKernels;

  // Counter to track the number of submissions of the kernel.
  // When this value is zero, it means that kernel is not submitted for an
  // execution - at this time we can release memory allocations referenced by
  // this kernel. We can do this when RefCount turns to 0 but it is too late
  // because kernels are cached in the context by SYCL RT and they are released
  // only during context object destruction. Regular RefCount is not usable to
  // track submissions because user/SYCL RT can retain kernel object any number
  // of times. And that's why there is no value of RefCount which can mean zero
  // submissions.
  std::atomic<uint32_t> SubmissionsCount;

  // Returns true if kernel has indirect access, false otherwise.
  bool hasIndirectAccess() {
    // Currently indirect access flag is set for all kernels and there is no API
    // to check if kernel actually indirectly access smth.
    return true;
  }

  // Hash function object for the unordered_set below.
  struct Hash {
    size_t operator()(const std::pair<void *const, MemAllocRecord> *P) const {
      return std::hash<void *>()(P->first);
    }
  };

  // If kernel has indirect access we need to make a snapshot of all existing
  // memory allocations to defer deletion of these memory allocations to the
  // moment when kernel execution has finished.
  // We store pointers to the elements because pointers are not invalidated by
  // insert/delete for std::unordered_map (iterators are invalidated). We need
  // to take a snapshot instead of just reference-counting the allocations,
  // because picture of active allocations can change during kernel execution
  // (new allocations can be added) and we need to know which memory allocations
  // were retained by this kernel to release them (and don't touch new
  // allocations) at kernel completion. Same kernel may be submitted several
  // times and retained allocations may be different at each submission. That's
  // why we have a set of memory allocations here and increase ref count only
  // once even if kernel is submitted many times. We don't want to know how many
  // times and which allocations were retained by each submission. We release
  // all allocations in the set only when SubmissionsCount == 0.
  std::unordered_set<std::pair<void *const, MemAllocRecord> *, Hash> MemAllocs;

  // Completed initialization of PI kernel. Must be called after construction.
  ur_result_t initialize();

  // Keeps info about an argument to the kernel enough to set it with
  // zeKernelSetArgumentValue.
  struct ArgumentInfo {
    uint32_t Index;
    size_t Size;
    // const ur_mem_handle_t_ *Value;
    ur_mem_handle_t_ *Value;
    ur_mem_handle_t_::access_mode_t AccessMode{ur_mem_handle_t_::unknown};
  };
  // Arguments that still need to be set (with zeKernelSetArgumentValue)
  // before kernel is enqueued.
  std::vector<ArgumentInfo> PendingArguments;

  // Cache of the kernel properties.
  ZeCache<ZeStruct<ze_kernel_properties_t>> ZeKernelProperties;
  ZeCache<std::string> ZeKernelName;
};
