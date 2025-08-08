//==----------- memory_pool_impl.hpp --- SYCL asynchronous allocation ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <sycl/context.hpp> // for context
#include <sycl/device.hpp>  // for device
#include <sycl/ext/oneapi/experimental/async_alloc/memory_pool.hpp>
#include <sycl/queue.hpp>         // for queue
#include <sycl/usm/usm_enums.hpp> // for usm::alloc

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {
namespace detail {

// Type to store pool properties values.
// Every property is represented by a pair that represent
// (is_property_assigned, property_value)
struct pool_properties {
  size_t initial_threshold;
  size_t maximum_size;
  bool zero_init;
};

class memory_pool_impl {
public:
  memory_pool_impl(const sycl::context &ctx, const sycl::device &dev,
                   const sycl::usm::alloc kind,
                   const pool_properties props = {});
  memory_pool_impl(const sycl::context &ctx, const sycl::device &dev,
                   const sycl::usm::alloc kind, ur_usm_pool_handle_t poolHandle,
                   const bool isDefaultPool, const pool_properties props = {});

  ~memory_pool_impl();

  memory_pool_impl(const memory_pool_impl &) = delete;
  memory_pool_impl &operator=(const memory_pool_impl &) = delete;

  ur_usm_pool_handle_t get_handle() const { return MPoolHandle; }
  sycl::device get_device() const { return MDevice; }
  sycl::context get_context() const {
    return sycl::detail::createSyclObjFromImpl<sycl::context>(MContextImplPtr);
  }
  sycl::usm::alloc get_alloc_kind() const { return MKind; }
  const pool_properties &getProps() const { return MProps; }

  // Returns backend specific values.
  size_t get_allocation_chunk_size() const;
  size_t get_threshold() const;
  size_t get_reserved_size_current() const;
  size_t get_reserved_size_high() const;
  size_t get_used_size_current() const;
  size_t get_used_size_high() const;

  void set_new_threshold(size_t newThreshold);
  void reset_reserved_size_high();
  void reset_used_size_high();

private:
  std::shared_ptr<sycl::detail::context_impl> MContextImplPtr;
  sycl::device MDevice;
  sycl::usm::alloc MKind;
  ur_usm_pool_handle_t MPoolHandle{0};
  bool MIsDefaultPool = false;
  pool_properties MProps;
};

} // namespace detail
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
