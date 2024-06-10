//==--------------- prefetch.hpp --- SYCL prefetch extension ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/__spirv/spirv_ops.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>
#include <sycl/types.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

enum class cache_level { L1 = 0, L2 = 1, L3 = 2, L4 = 3 };

struct nontemporal;

struct prefetch_hint_key {
  template <cache_level Level, typename Hint>
  using value_t =
      property_value<prefetch_hint_key,
                     std::integral_constant<cache_level, Level>, Hint>;
};

template <cache_level Level, typename Hint>
inline constexpr prefetch_hint_key::value_t<Level, Hint> prefetch_hint;

inline constexpr prefetch_hint_key::value_t<cache_level::L1, void>
    prefetch_hint_L1;
inline constexpr prefetch_hint_key::value_t<cache_level::L2, void>
    prefetch_hint_L2;
inline constexpr prefetch_hint_key::value_t<cache_level::L3, void>
    prefetch_hint_L3;
inline constexpr prefetch_hint_key::value_t<cache_level::L4, void>
    prefetch_hint_L4;

inline constexpr prefetch_hint_key::value_t<cache_level::L1, nontemporal>
    prefetch_hint_L1_nt;
inline constexpr prefetch_hint_key::value_t<cache_level::L2, nontemporal>
    prefetch_hint_L2_nt;
inline constexpr prefetch_hint_key::value_t<cache_level::L3, nontemporal>
    prefetch_hint_L3_nt;
inline constexpr prefetch_hint_key::value_t<cache_level::L4, nontemporal>
    prefetch_hint_L4_nt;

namespace detail {
template <> struct IsCompileTimeProperty<prefetch_hint_key> : std::true_type {};

template <cache_level Level, typename Hint>
struct PropertyMetaInfo<prefetch_hint_key::value_t<Level, Hint>> {
  static constexpr const char *name = std::is_same_v<Hint, nontemporal>
                                          ? "sycl-prefetch-hint-nt"
                                          : "sycl-prefetch-hint";
  static constexpr int value = static_cast<int>(Level);
};

template <access::address_space AS>
inline constexpr bool check_prefetch_AS =
    AS == access::address_space::global_space ||
    AS == access::address_space::generic_space;

template <access_mode mode>
inline constexpr bool check_prefetch_acc_mode =
    mode == access_mode::read || mode == access_mode::read_write;

template <typename T, typename Properties>
void prefetch_impl(T *ptr, size_t bytes, Properties properties) {
#ifdef __SYCL_DEVICE_ONLY__
  auto *ptrGlobalAS = __SYCL_GenericCastToPtrExplicit_ToGlobal<const char>(ptr);
  const __attribute__((opencl_global)) char *ptrAnnotated = nullptr;
  if constexpr (!properties.template has_property<prefetch_hint_key>()) {
    ptrAnnotated = __builtin_intel_sycl_ptr_annotation(
        ptrGlobalAS, "sycl-prefetch-hint", static_cast<int>(cache_level::L1));
  } else {
    auto prop = properties.template get_property<prefetch_hint_key>();
    ptrAnnotated = __builtin_intel_sycl_ptr_annotation(
        ptrGlobalAS, PropertyMetaInfo<decltype(prop)>::name,
        PropertyMetaInfo<decltype(prop)>::value);
  }
  __spirv_ocl_prefetch(ptrAnnotated, bytes);
#else
  std::ignore = ptr;
  std::ignore = bytes;
  std::ignore = properties;
#endif
}

template <typename Group, typename T, typename Properties>
void joint_prefetch_impl(Group g, T *ptr, size_t bytes, Properties properties) {
  // Although calling joint_prefetch is functionally equivalent to calling
  // prefetch from every work-item in a group, native suppurt may be added to to
  // issue cooperative prefetches more efficiently on some hardware.
  std::ignore = g;
  prefetch_impl(ptr, bytes, properties);
}
} // namespace detail

template <typename Properties = empty_properties_t>
std::enable_if_t<is_property_list_v<std::decay_t<Properties>>>
prefetch(void *ptr, Properties properties = {}) {
  detail::prefetch_impl(ptr, 1, properties);
}

template <typename Properties = empty_properties_t>
std::enable_if_t<is_property_list_v<std::decay_t<Properties>>>
prefetch(void *ptr, size_t bytes, Properties properties = {}) {
  detail::prefetch_impl(ptr, bytes, properties);
}

template <typename T, typename Properties = empty_properties_t>
std::enable_if_t<is_property_list_v<std::decay_t<Properties>>>
prefetch(T *ptr, Properties properties = {}) {
  detail::prefetch_impl(ptr, sizeof(T), properties);
}

template <typename T, typename Properties = empty_properties_t>
std::enable_if_t<is_property_list_v<std::decay_t<Properties>>>
prefetch(T *ptr, size_t count, Properties properties = {}) {
  detail::prefetch_impl(ptr, count * sizeof(T), properties);
}

template <access::address_space AddressSpace, access::decorated IsDecorated,
          typename Properties = empty_properties_t>
std::enable_if_t<detail::check_prefetch_AS<AddressSpace> &&
                 is_property_list_v<std::decay_t<Properties>>>
prefetch(multi_ptr<void, AddressSpace, IsDecorated> ptr,
         Properties properties = {}) {
  detail::prefetch_impl(ptr.get(), 1, properties);
}

template <access::address_space AddressSpace, access::decorated IsDecorated,
          typename Properties = empty_properties_t>
std::enable_if_t<detail::check_prefetch_AS<AddressSpace> &&
                 is_property_list_v<std::decay_t<Properties>>>
prefetch(multi_ptr<void, AddressSpace, IsDecorated> ptr, size_t bytes,
         Properties properties = {}) {
  detail::prefetch_impl(ptr.get(), bytes, properties);
}

template <typename T, access::address_space AddressSpace,
          access::decorated IsDecorated,
          typename Properties = empty_properties_t>
std::enable_if_t<detail::check_prefetch_AS<AddressSpace> &&
                 is_property_list_v<std::decay_t<Properties>>>
prefetch(multi_ptr<T, AddressSpace, IsDecorated> ptr,
         Properties properties = {}) {
  detail::prefetch_impl(ptr.get(), sizeof(T), properties);
}

template <typename T, access::address_space AddressSpace,
          access::decorated IsDecorated,
          typename Properties = empty_properties_t>
std::enable_if_t<detail::check_prefetch_AS<AddressSpace> &&
                 is_property_list_v<std::decay_t<Properties>>>
prefetch(multi_ptr<T, AddressSpace, IsDecorated> ptr, size_t count,
         Properties properties = {}) {
  detail::prefetch_impl(ptr.get(), count * sizeof(T), properties);
}

template <typename DataT, int Dimensions, access_mode AccessMode,
          access::placeholder IsPlaceholder,
          typename Properties = empty_properties_t,
          typename AccessorProperties = empty_properties_t>
std::enable_if_t<detail::check_prefetch_acc_mode<AccessMode> &&
                 (Dimensions > 0) &&
                 is_property_list_v<std::decay_t<Properties>>>
prefetch(accessor<DataT, Dimensions, AccessMode, target::device, IsPlaceholder,
                  AccessorProperties>
             acc,
         id<Dimensions> offset, Properties properties = {}) {
  detail::prefetch_impl(&acc[offset], sizeof(DataT), properties);
}

template <typename DataT, int Dimensions, access_mode AccessMode,
          access::placeholder IsPlaceholder,
          typename Properties = empty_properties_t,
          typename AccessorProperties = empty_properties_t>
std::enable_if_t<detail::check_prefetch_acc_mode<AccessMode> &&
                 (Dimensions > 0) &&
                 is_property_list_v<std::decay_t<Properties>>>
prefetch(accessor<DataT, Dimensions, AccessMode, target::device, IsPlaceholder,
                  AccessorProperties>
             acc,
         size_t offset, size_t count, Properties properties = {}) {
  detail::prefetch_impl(&acc[offset], count * sizeof(DataT), properties);
}

template <typename Group, typename Properties = empty_properties_t>
std::enable_if_t<sycl::is_group_v<std::decay_t<Group>> &&
                 is_property_list_v<std::decay_t<Properties>>>
joint_prefetch(Group g, void *ptr, Properties properties = {}) {
  detail::joint_prefetch_impl(g, ptr, 1, properties);
}

template <typename Group, typename Properties = empty_properties_t>
std::enable_if_t<sycl::is_group_v<std::decay_t<Group>> &&
                 is_property_list_v<std::decay_t<Properties>>>
joint_prefetch(Group g, void *ptr, size_t bytes, Properties properties = {}) {
  detail::joint_prefetch_impl(g, ptr, bytes, properties);
}

template <typename Group, typename T, typename Properties = empty_properties_t>
std::enable_if_t<sycl::is_group_v<std::decay_t<Group>> &&
                 is_property_list_v<std::decay_t<Properties>>>
joint_prefetch(Group g, T *ptr, Properties properties = {}) {
  detail::joint_prefetch_impl(g, ptr, sizeof(T), properties);
}

template <typename Group, typename T, typename Properties = empty_properties_t>
std::enable_if_t<sycl::is_group_v<std::decay_t<Group>> &&
                 is_property_list_v<std::decay_t<Properties>>>
joint_prefetch(Group g, T *ptr, size_t count, Properties properties = {}) {
  detail::joint_prefetch_impl(g, ptr, count * sizeof(T), properties);
}

template <typename Group, access::address_space AddressSpace,
          access::decorated IsDecorated,
          typename Properties = empty_properties_t>
std::enable_if_t<detail::check_prefetch_AS<AddressSpace> &&
                 sycl::is_group_v<std::decay_t<Group>> &&
                 is_property_list_v<std::decay_t<Properties>>>
joint_prefetch(Group g, multi_ptr<void, AddressSpace, IsDecorated> ptr,
               Properties properties = {}) {
  detail::joint_prefetch_impl(g, ptr.get(), 1, properties);
}

template <typename Group, access::address_space AddressSpace,
          access::decorated IsDecorated,
          typename Properties = empty_properties_t>
std::enable_if_t<detail::check_prefetch_AS<AddressSpace> &&
                 sycl::is_group_v<std::decay_t<Group>> &&
                 is_property_list_v<std::decay_t<Properties>>>
joint_prefetch(Group g, multi_ptr<void, AddressSpace, IsDecorated> ptr,
               size_t bytes, Properties properties = {}) {
  detail::joint_prefetch_impl(g, ptr.get(), bytes, properties);
}

template <typename Group, typename T, access::address_space AddressSpace,
          access::decorated IsDecorated,
          typename Properties = empty_properties_t>
std::enable_if_t<detail::check_prefetch_AS<AddressSpace> &&
                 sycl::is_group_v<std::decay_t<Group>> &&
                 is_property_list_v<std::decay_t<Properties>>>
joint_prefetch(Group g, multi_ptr<T, AddressSpace, IsDecorated> ptr,
               Properties properties = {}) {
  detail::joint_prefetch_impl(g, ptr.get(), sizeof(T), properties);
}

template <typename Group, typename T, access::address_space AddressSpace,
          access::decorated IsDecorated,
          typename Properties = empty_properties_t>
std::enable_if_t<detail::check_prefetch_AS<AddressSpace> &&
                 sycl::is_group_v<std::decay_t<Group>> &&
                 is_property_list_v<std::decay_t<Properties>>>
joint_prefetch(Group g, multi_ptr<T, AddressSpace, IsDecorated> ptr,
               size_t count, Properties properties = {}) {
  detail::joint_prefetch_impl(g, ptr.get(), count * sizeof(T), properties);
}

template <typename Group, typename DataT, int Dimensions,
          access_mode AccessMode, access::placeholder IsPlaceholder,
          typename Properties = empty_properties_t,
          typename AccessorProperties = empty_properties_t>
std::enable_if_t<detail::check_prefetch_acc_mode<AccessMode> &&
                 (Dimensions > 0) && sycl::is_group_v<std::decay_t<Group>> &&
                 is_property_list_v<std::decay_t<Properties>>>
joint_prefetch(Group g,
               accessor<DataT, Dimensions, AccessMode, target::device,
                        IsPlaceholder, AccessorProperties>
                   acc,
               size_t offset, Properties properties = {}) {
  detail::joint_prefetch_impl(g, &acc[offset], sizeof(DataT), properties);
}

template <typename Group, typename DataT, int Dimensions,
          access_mode AccessMode, access::placeholder IsPlaceholder,
          typename Properties = empty_properties_t,
          typename AccessorProperties = empty_properties_t>
std::enable_if_t<detail::check_prefetch_acc_mode<AccessMode> &&
                 (Dimensions > 0) && sycl::is_group_v<std::decay_t<Group>> &&
                 is_property_list_v<std::decay_t<Properties>>>
joint_prefetch(Group g,
               accessor<DataT, Dimensions, AccessMode, target::device,
                        IsPlaceholder, AccessorProperties>
                   acc,
               size_t offset, size_t count, Properties properties = {}) {
  detail::joint_prefetch_impl(g, &acc[offset], count * sizeof(DataT),
                              properties);
}

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
