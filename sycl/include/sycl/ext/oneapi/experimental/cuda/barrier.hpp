//==--- barrier.hpp - SYCL_ONEAPI_BARRIER  ---------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/__spirv/spirv_ops.hpp> // for __clc_BarrierInitialize
#include <sycl/exception.hpp>

#include <stdint.h> // for int32_t, int64_t, uint32_t, uint64_t

#define SYCL_EXT_ONEAPI_CUDA_ASYNC_BARRIER 1

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace oneapi {
namespace experimental {
namespace cuda {

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
    (void)state;
    (void)expected_count;
    throw exception(make_error_code(errc::runtime),
                    "Barrier is not supported on host.");
#endif
  }

  void invalidate() {
#ifdef __SYCL_DEVICE_ONLY__
    __clc_BarrierInvalidate(&state);
#else
    throw exception(make_error_code(errc::runtime),
                    "Barrier is not supported on host.");
#endif
  }

  arrival_token arrive() {
#ifdef __SYCL_DEVICE_ONLY__
    return __clc_BarrierArrive(&state);
#else
    throw exception(make_error_code(errc::runtime),
                    "Barrier is not supported on host.");
#endif
  }

  arrival_token arrive_and_drop() {
#ifdef __SYCL_DEVICE_ONLY__
    return __clc_BarrierArriveAndDrop(&state);
#else
    throw exception(make_error_code(errc::runtime),
                    "Barrier is not supported on host.");
#endif
  }

  arrival_token arrive_no_complete(int32_t count) {
#ifdef __SYCL_DEVICE_ONLY__
    return __clc_BarrierArriveNoComplete(&state, count);
#else
    (void)count;
    throw exception(make_error_code(errc::runtime),
                    "Barrier is not supported on host.");
#endif
  }

  arrival_token arrive_and_drop_no_complete(int32_t count) {
#ifdef __SYCL_DEVICE_ONLY__
    return __clc_BarrierArriveAndDropNoComplete(&state, count);
#else
    (void)count;
    throw exception(make_error_code(errc::runtime),
                    "Barrier is not supported on host.");
#endif
  }

  void arrive_copy_async() {
#ifdef __SYCL_DEVICE_ONLY__
    __clc_BarrierCopyAsyncArrive(&state);
#else
    throw exception(make_error_code(errc::runtime),
                    "Barrier is not supported on host.");
#endif
  }

  void arrive_copy_async_no_inc() {
#ifdef __SYCL_DEVICE_ONLY__
    __clc_BarrierCopyAsyncArriveNoInc(&state);
#else
    throw exception(make_error_code(errc::runtime),
                    "Barrier is not supported on host.");
#endif
  }

  void wait(arrival_token arrival) {
#ifdef __SYCL_DEVICE_ONLY__
    __clc_BarrierWait(&state, arrival);
#else
    (void)arrival;
    throw exception(make_error_code(errc::runtime),
                    "Barrier is not supported on host.");
#endif
  }

  bool test_wait(arrival_token arrival) {
#ifdef __SYCL_DEVICE_ONLY__
    return __clc_BarrierTestWait(&state, arrival);
#else
    (void)arrival;
    throw exception(make_error_code(errc::runtime),
                    "Barrier is not supported on host.");
#endif
  }

  void arrive_and_wait() {
#ifdef __SYCL_DEVICE_ONLY__
    __clc_BarrierArriveAndWait(&state);
#else
    throw exception(make_error_code(errc::runtime),
                    "Barrier is not supported on host.");
#endif
  }

// On Windows certain headers define macros min/max
#pragma push_macro("max")
#ifdef max
#undef max
#endif
  static constexpr uint64_t max() { return (1 << 20) - 1; }
#pragma pop_macro("max")
};
} // namespace cuda
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
