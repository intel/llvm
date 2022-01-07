//==--- barrier.hpp - SYCL_ONEAPI_BARRIER  ---------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/__spirv/spirv_ops.hpp>
#include <cstddef>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl::ext::oneapi {

class barrier {
  int64_t state;

public:
  using arrival_token = int64_t;

  // barriers cannot be moved or copied
  barrier(const barrier &other) = delete;
  barrier(barrier &&other) noexcept = delete;
  barrier &operator=(const barrier &other) = delete;
  barrier &operator=(barrier &&other) noexcept = delete;

  // the valid range of count for PTX is [1,2^20 -1] -
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-contents
  // expected to sync after this so all threads see the initialized barrier
  // (maybe the sync should be part of the constructor)
  // TODO only one thread needs to initialize, but many init is fine
  void initialize(uint32_t expected_count) {
#ifdef __SYCL_DEVICE_ONLY__
    __spirv_BarrierInitialize(&state, expected_count);
#else
    throw runtime_error("Barrier is not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  void invalidate() {
#ifdef __SYCL_DEVICE_ONLY__
    __spirv_BarrierInvalidate(&state);
#else
    throw runtime_error("Barrier is not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  arrival_token arrive() {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_BarrierArrive(&state);
#else
    throw runtime_error("Barrier is not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  void wait(arrival_token arrival) {
#ifdef __SYCL_DEVICE_ONLY__
    __spirv_BarrierWait(&state, arrival);
#else
    throw runtime_error("Barrier is not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  /*
  // equivalent to wait(arrive());
  void arrive_and_wait();
  // arrive and also drop the expected count by one
  void arrive_and_drop();


  // returns the maximum value of expected count that is supported by the
  implementation. static constexpr std::uint32_t max() noexcept();
  */
};

} // namespace sycl::ext::oneapi
} // __SYCL_INLINE_NAMESPACE(cl)