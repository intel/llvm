//==--- barrier.hpp - SYCL_ONEAPI_BARRIER  ---------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#define SYCL_EXT_ONEAPI_CUDA_ASYNC_BARRIER 1

#include <CL/__spirv/spirv_ops.hpp>
#include <cstddef>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl::ext::oneapi::experimental::cuda {

class barrier {
  int64_t state;

public:
  using arrival_token = int64_t;

  // barriers cannot be moved or copied
  barrier(const barrier &other) = delete;
  barrier(barrier &&other) noexcept = delete;
  barrier &operator=(const barrier &other) = delete;
  barrier &operator=(barrier &&other) noexcept = delete;

  void initialize(uint32_t expected_count) {
#ifdef __SYCL_DEVICE_ONLY__
    __clc_BarrierInitialize(&state, expected_count);
#else
    throw runtime_error("Barrier is not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  void invalidate() {
#ifdef __SYCL_DEVICE_ONLY__
    __clc_BarrierInvalidate(&state);
#else
    throw runtime_error("Barrier is not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  arrival_token arrive() {
#ifdef __SYCL_DEVICE_ONLY__
    return __clc_BarrierArrive(&state);
#else
    throw runtime_error("Barrier is not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  arrival_token arrive_and_drop() {
#ifdef __SYCL_DEVICE_ONLY__
    return __clc_BarrierArriveAndDrop(&state);
#else
    throw runtime_error("Barrier is not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  arrival_token arrive_no_complete(int32_t count) {
#ifdef __SYCL_DEVICE_ONLY__
    return __clc_BarrierArriveNoComplete(&state, count);
#else
    throw runtime_error("Barrier is not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  arrival_token arrive_and_drop_no_complete(int32_t count) {
#ifdef __SYCL_DEVICE_ONLY__
    return __clc_BarrierArriveAndDropNoComplete(&state, count);
#else
    throw runtime_error("Barrier is not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  void arrive_copy_async() {
#ifdef __SYCL_DEVICE_ONLY__
    __clc_BarrierCopyAsyncArrive(&state);
#else
    throw runtime_error("Barrier is not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  void arrive_copy_async_no_inc() {
#ifdef __SYCL_DEVICE_ONLY__
    __clc_BarrierCopyAsyncArriveNoInc(&state);
#else
    throw runtime_error("Barrier is not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  void wait(arrival_token arrival) {
#ifdef __SYCL_DEVICE_ONLY__
    __clc_BarrierWait(&state, arrival);
#else
    throw runtime_error("Barrier is not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  bool test_wait(arrival_token arrival) {
#ifdef __SYCL_DEVICE_ONLY__
    return __clc_BarrierTestWait(&state, arrival);
#else
    throw runtime_error("Barrier is not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  void arrive_and_wait() {
#ifdef __SYCL_DEVICE_ONLY__
    __clc_BarrierArriveAndWait(&state);
#else
    throw runtime_error("Barrier is not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  static constexpr uint64_t max() { return (1 << 20) - 1; }
};

} // namespace sycl::ext::oneapi::cuda
} // __SYCL_INLINE_NAMESPACE(cl)
