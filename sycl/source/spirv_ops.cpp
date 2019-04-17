//===------------- spirv_ops.cpp - SPIRV operations -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/__spirv/spirv_ops.hpp>
#include <CL/sycl/exception.hpp>
#include <atomic>

namespace cl {
namespace __spirv {

// This operation is NOP on HOST as all operations there are blocking and
// by the moment this function was called, the operations generating
// the OpTypeEvent objects had already been finished.
void OpGroupWaitEvents(int32_t Scope, uint32_t NumEvents,
                              OpTypeEvent ** WaitEvents) noexcept {
}

void OpControlBarrier(Scope Execution, Scope Memory,
                      uint32_t Semantics) noexcept {
  throw cl::sycl::runtime_error(
      "Barrier is not supported on the host device yet.");
}

void OpMemoryBarrier(Scope Memory, uint32_t Semantics) noexcept {
  // 1. The 'Memory' parameter is ignored on HOST because there is no memory
  //    separation to global and local there.
  // 2. The 'Semantics' parameter is ignored because there is no need
  //    to distinguish the classes of memory (workgroup/cross-workgroup/etc).
  atomic_thread_fence(std::memory_order_seq_cst);
}

void prefetch(const char *Ptr, size_t NumBytes) noexcept {
  // TODO: the cache line size may be different.
  const size_t CacheLineSize = 64;
  size_t NumCacheLines =
      (NumBytes / CacheLineSize) + ((NumBytes % CacheLineSize) ? 1 : 0);
  for (; NumCacheLines != 0; NumCacheLines--) {
    __builtin_prefetch(reinterpret_cast<const void *>(Ptr));
    Ptr += 64;
  }
}

} // namespace __spirv
} // namespace cl
