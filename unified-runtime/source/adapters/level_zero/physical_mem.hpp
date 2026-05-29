//===---------------- physical_mem.hpp - Level Zero Adapter ---------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "common.hpp"
#include "common/ur_ref_count.hpp"

// Opaque handle data exchanged between processes for physical memory IPC.
// Contains the IPC handle obtained from zeMemGetIpcHandleWithProperties
// plus the allocation size required to reconstruct the physical_mem on import.
// When the driver correctly implements ZE_IPC_MEM_HANDLE_TYPE_FLAG_DEFAULT for
// physical memory, ze_ipc_mem_handle_t is a self-contained 64-byte opaque blob
// that can be transferred to another process without fd passing.  On drivers
// that fall back to an fd-based handle, urIPCGetPhysMemHandleExp detects and
// rejects the fd handle, returning UR_RESULT_ERROR_UNSUPPORTED_FEATURE.
struct ZeIPCPhysMemHandleData {
  ze_ipc_mem_handle_t IpcHandle; // opaque IPC handle (64 bytes)
  size_t Size;
};

struct ur_physical_mem_handle_t_ : ur_object {
  ur_physical_mem_handle_t_(ze_physical_mem_handle_t ZePhysicalMem,
                            ur_context_handle_t Context,
                            ur_device_handle_t Device, size_t Size,
                            bool EnableIpc, void *IpcVirtualAddress = nullptr)
      : ZePhysicalMem{ZePhysicalMem}, Context{Context}, Device{Device},
        Size{Size}, EnableIpc{EnableIpc}, IpcVirtualAddress{IpcVirtualAddress} {
  }

  // Level Zero physical memory handle. Null for IPC-opened handles (consumer
  // side), where the memory is accessible via IpcVirtualAddress instead.
  ze_physical_mem_handle_t ZePhysicalMem;

  // Keeps the PI context of this memory handle.
  ur_context_handle_t Context;

  // Device this physical memory was allocated on.
  ur_device_handle_t Device;

  // Size in bytes of this physical memory allocation.
  size_t Size;

  // Whether this allocation was created with IPC export enabled.
  bool EnableIpc;

  // Virtual address returned by zeMemOpenIpcHandle for IPC-opened handles
  // (consumer side). Non-null means the memory is already virtually mapped
  // at this address; cleanup uses zeMemCloseIpcHandle rather than
  // zePhysicalMemDestroy.
  void *IpcVirtualAddress;

  ur::RefCount RefCount;
};
