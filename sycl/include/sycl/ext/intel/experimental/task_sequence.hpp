//==--------------------------- task_sequence.hpp --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/__spirv/spirv_ops.hpp>
#include <CL/__spirv/spirv_types.hpp>
#include <sycl/exception.hpp>
#include <sycl/ext/intel/experimental/fpga_kernel_properties.hpp>
#include <sycl/ext/intel/experimental/task_sequence_properties.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>

#include <type_traits>

namespace sycl {
inline namespace _V1 {
namespace ext::intel::experimental {

template <auto &f> struct task_sequence_checker : std::false_type {};

template <typename ReturnT, typename... ArgsT, ReturnT (&f)(ArgsT...)>
struct task_sequence_checker<f> : std::true_type {};

template <auto &f,
          typename PropertyListT = oneapi::experimental::empty_properties_t>
class task_sequence {

  static_assert(task_sequence_checker<f>::value);

  static_assert(oneapi::experimental::is_property_list<PropertyListT>::value,
                "Property list is invalid.");
};

template <typename ReturnT, typename... ArgsT, ReturnT (&f)(ArgsT...),
          typename... Props>
#if defined(__SYCL_DEVICE_ONLY__)
class
    [[__sycl_detail__::__uses_aspects__(aspect::ext_intel_fpga_task_sequence)]] task_sequence<
#else
class task_sequence<
#endif
        f, oneapi::experimental::detail::properties_t<Props...>> {
  using property_list_t = oneapi::experimental::detail::properties_t<Props...>;

public:
  task_sequence(const task_sequence &) = delete;
  task_sequence &operator=(const task_sequence &) = delete;
  task_sequence(task_sequence &&) = delete;
  task_sequence &operator=(task_sequence &&) = delete;

  task_sequence() {
#if defined(__SYCL_DEVICE_ONLY__)
    taskSequence = __spirv_TaskSequenceCreateINTEL(
        &f, pipelined, fpga_cluster, response_capacity, invocation_capacity);
#else
    throw exception{make_error_code(errc::feature_not_supported),
                    "task_sequence is not supported on the host"};
#endif
  }

  void async([[maybe_unused]] ArgsT... Args) {
#if defined(__SYCL_DEVICE_ONLY__)
    ++outstanding;
    __spirv_TaskSequenceAsyncINTEL(taskSequence, Args...);
#else
    throw exception{make_error_code(errc::feature_not_supported),
                    "task_sequence is not supported on the host"};
#endif
  }

  ReturnT get() {
#if defined(__SYCL_DEVICE_ONLY__)
    --outstanding;
    return __spirv_TaskSequenceGetINTEL<ReturnT>(taskSequence);
#else
    throw exception{make_error_code(errc::feature_not_supported),
                    "task_sequence is not supported on the host"};
#endif
  }

  ~task_sequence() {
#if defined(__SYCL_DEVICE_ONLY__)
    if constexpr (!has_property<balanced_key>()) {
      while (outstanding > 0)
        get();
    }
    __spirv_TaskSequenceReleaseINTEL(taskSequence);
#endif
    // Destructor shouldn't throw exception.
  }

  template <typename propertyT> static constexpr bool has_property() {
    return property_list_t::template has_property<propertyT>();
  }

  template <typename propertyT> static constexpr auto get_property() {
    return property_list_t::template get_property<propertyT>();
  }

private:
#if defined(__SYCL_DEVICE_ONLY__)
  unsigned outstanding = 0;
  __spv::__spirv_TaskSequenceINTEL *taskSequence;
#endif
  static constexpr int32_t pipelined =
      oneapi::experimental::detail::ValueOrDefault<
          property_list_t, pipelined_key>::template get<int32_t>(-1);
  static constexpr int32_t fpga_cluster =
      has_property<fpga_cluster_key>()
          ? static_cast<
                typename std::underlying_type<fpga_cluster_options_enum>::type>(
                oneapi::experimental::detail::ValueOrDefault<property_list_t,
                                                             fpga_cluster_key>::
                    template get<fpga_cluster_options_enum>(
                        fpga_cluster_options_enum::stall_free))
          : -1;
  static constexpr uint32_t response_capacity =
      oneapi::experimental::detail::ValueOrDefault<
          property_list_t, response_capacity_key>::template get<uint32_t>(0);
  static constexpr uint32_t invocation_capacity =
      oneapi::experimental::detail::ValueOrDefault<
          property_list_t, invocation_capacity_key>::template get<uint32_t>(0);
};

} // namespace ext::intel::experimental
} // namespace _V1
} // namespace sycl
