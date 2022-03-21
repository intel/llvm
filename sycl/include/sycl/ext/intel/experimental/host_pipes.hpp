//==---------------- pipes.hpp - SYCL pipes ------------*- C++ -*-----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include <CL/sycl/context.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/queue.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>
#include <sycl/ext/oneapi/properties/property.hpp>
#include <sycl/ext/oneapi/properties/property_utils.hpp>
#include <sycl/ext/oneapi/properties/property_value.hpp>

#ifdef XPTI_ENABLE_INSTRUMENTATION
#include <xpti/xpti_data_types.h>
#include <xpti/xpti_trace_framework.hpp>
#endif

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace oneapi {
namespace experimental {

// min_capacity property has one integer non-type parameter.
struct min_capacity_key {
  template <int capacity>
  using value_t =
      property_value<min_capacity_key, std::integral_constant<int, capacity>>;
};
// min_capacity is an object of a property value type of min_capacity.
template <int capacity>
inline constexpr min_capacity_key::value_t<capacity> min_capacity;

template <> struct is_property_key<min_capacity_key> : std::true_type {};

namespace detail {

template <> struct PropertyToKind<min_capacity_key> {
  static constexpr PropKind Kind = PropKind::MinCapacity;
};

template <> struct IsCompileTimeProperty<min_capacity_key> : std::true_type {};

} // namespace detail

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace intel {
namespace experimental {

using default_pipe_properties =
    decltype(sycl::ext::oneapi::experimental::properties(
        sycl::ext::oneapi::experimental::min_capacity<0>));

template <class _name, typename _dataT,
          typename PropertyList = default_pipe_properties>
class
#ifdef __SYCL_DEVICE_ONLY__
    [[__sycl_detail__::add_ir_attributes_global_variable("sycl-host-access",
                                                         "readwrite")]]
#endif
    host_pipe { // TODO change name to pipe, and merge into the existing pipe
                // implementation
  static_assert(
      sycl::ext::oneapi::experimental::is_property_list_v<PropertyList>,
      "Host pipe is available only through new property list");

public:
  using value_type = _dataT;
  static constexpr int32_t min_cap =
      PropertyList::template has_property<
          sycl::ext::oneapi::experimental::min_capacity_key>()
          ? PropertyList::template get_property<
                sycl::ext::oneapi::experimental::min_capacity_key>()
                .value
          : 0;

  // Blocking pipes
  static _dataT read(queue & q, memory_order order = memory_order::seq_cst);
  static void write(queue & q, const _dataT &data,
                    memory_order order = memory_order::seq_cst);
  // Non-blocking pipes
  static _dataT read(queue & q, bool &success_code,
                     memory_order order = memory_order::seq_cst);
  static void write(queue & q, const _dataT &data, bool &success_code,
                    memory_order order = memory_order::seq_cst);

private:
  static constexpr int32_t m_Size = sizeof(_dataT);
  static constexpr int32_t m_Alignment = alignof(_dataT);
  static constexpr int32_t ID = _name::id;
#ifdef __SYCL_DEVICE_ONLY__
  static constexpr struct ConstantPipeStorage m_Storage
      __attribute__((io_pipe_id(ID))) = {m_Size, m_Alignment, min_capacity};
#endif // __SYCL_DEVICE_ONLY__
};

} // namespace experimental
} // namespace intel
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
