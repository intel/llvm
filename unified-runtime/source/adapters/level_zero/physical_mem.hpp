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

// Discriminates between the two IPC physical-memory sharing paths.
enum class ZeIPCPhysMemHandleKind : uint32_t {
  // BMG and newer (Xe2+): opaque 64-byte IPC handle obtained from
  // zeMemGetIpcHandleWithProperties.  Can be transferred as raw bytes.
  OpaqueBMGOrNewer = 0,
  // Pre-BMG (DG2, PVC, MTL, ARL): Linux DMA-BUF file descriptor obtained
  // via zePhysicalMemGetProperties.  The importing process duplicates the fd
  // with pidfd_getfd(2) (Linux 5.6+) using the exporting process's PID and fd.
  DmaBufFd = 1,
};

// Opaque handle data exchanged between processes for physical memory IPC.
// Supports two paths discriminated by Kind (always zero-initialize before use):
//  - OpaqueBMGOrNewer: self-contained blob; IpcHandle is valid.
//  - DmaBufFd: Pid/Fd are valid (Linux only); Fd stays open until
//    urIPCPutPhysMemHandleExp is called by the exporter.
struct ZeIPCPhysMemHandleData {
  ZeIPCPhysMemHandleKind Kind;
  size_t Size;
  ze_ipc_mem_handle_t IpcHandle; // valid when Kind == OpaqueBMGOrNewer
#ifdef __linux__
  pid_t Pid; // valid when Kind == DmaBufFd: PID of the exporting process
  int Fd;    // valid when Kind == DmaBufFd: DMA-BUF fd in exporting process
#endif
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
