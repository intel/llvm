//===---------- physical_mem.hpp - Level Zero Adapter v2 ----------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "../physical_mem.hpp"

// Opaque handle data exchanged between processes for physical memory IPC.
// Contains the PID and file descriptor exported by zePhysicalMemGetProperties,
// plus the allocation size required by zePhysicalMemCreate on import.
// Cross-process access uses pidfd_getfd(2) (Linux 5.6+): the consumer obtains
// a duplicate of the spawner's DMA-BUF fd via the spawner's PID.
// Only defined on Linux because pid_t and DMA-BUF fds are Linux concepts.
#ifdef __linux__
struct ZeIPCPhysMemHandleData {
  pid_t Pid; // PID of the exporting process
  int Fd;    // DMA-BUF fd in the exporting process
  size_t Size;
};
#endif // __linux__
