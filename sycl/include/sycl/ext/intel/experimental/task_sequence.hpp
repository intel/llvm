//==--------------------------- task_sequence.hpp --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/__spirv/spirv_ops.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace intel {
namespace experimental {

template <auto &f, typename PropertyListT>
struct task_sequence;

template<auto &f> struct task_sequence_checker : std::false_type {};

template<typename ReturnT, typename... ArgsT, ReturnT (&f)(ArgsT...)>
struct task_sequence_checker<f> : std::true_type {};

template<auto& f, class propertiesT =
  decltype(oneapi::experimental::properties{})>
class task_sequence{ static_assert(task_sequence_checker<f>::value); };

template <typename ReturnT, typename... ArgsT, ReturnT (&f)(ArgsT...),
          typename PropertyListT>
class task_sequence {
  // TODO: put atomic lock on it if required
  unsigned outstanding = 0;

  typedef ReturnT (*f_t)(ArgsT...);

public:
  task_sequence(const task_sequence &) = delete;
  task_sequence &operator=(const task_sequence &) = delete;
  task_sequence(task_sequence &&) = delete;
  task_sequence &operator=(task_sequence &&) = delete;

  template <typename PropertyT> static constexpr bool has_property() {
    return property_list_t::template has_property<PropertyT>();
  }

  template <typename PropertyT> static constexpr auto get_property() {
    return property_list_t::template get_property<PropertyT>();
  }

  task_sequence() {
#if defined(__SYCL_DEVICE_ONLY__)
    __spirv_TaskSequenceCreateINTEL(this, &f,
      (PropertyListT::template has_property<pipelined_key> ?
        PropertyListT::template get_property<pipelined_key>.value() : 1),
      (PropertyListT::template has_property<use_stall_enable_clusters_key> ?
        PropertyListT::template get_property<
          use_stall_enable_clusters_key>.value() : 1)
      );
#else
    throw exception{errc::feature_not_supported,
                    "task_sequence is not supported on host device"};
#endif
  }

  void async(ArgsT... Args) {
#if defined(__SYCL_DEVICE_ONLY__)
    ++outstanding;
    __spirv_TaskSequenceAsyncINTEL(this, PropertyListT::template get_property<
      invocation_capacity_key>, Args...);
#else
    throw exception{errc::feature_not_supported,
                    "task_sequence is not supported on host device"};
#endif
  }

  ReturnT get() {
#if defined(__SYCL_DEVICE_ONLY__)
    --outstanding;
    return __spirv_TaskSequenceGetINTEL(this,
      PropertyListT::template get_property<response_capacity_key>);
#else
    throw exception{errc::feature_not_supported,
                    "task_sequence is not supported on host device"};
#endif
  }

  ~task_sequence() {
#if defined(__SYCL_DEVICE_ONLY__)
    while (outstanding)
      get();
    __spirv_TaskSequenceReleaseINTEL(this);
#else
    // "task_sequence is not supported on host device"
    // Destructor shouldn't throw exception.
#endif
  }
};

} // namespace experimental
} // namespace intel
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
/* end INTEL_CUSTOMIZATION */
