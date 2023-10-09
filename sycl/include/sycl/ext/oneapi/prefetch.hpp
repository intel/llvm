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

template <typename Properties>
void prefetch_impl(void *ptr, size_t bytes, Properties properties) {
#ifdef __SYCL_DEVICE_ONLY__
  auto *ptrGlobalAS = __SYCL_GenericCastToPtrExplicit_ToGlobal<char>(ptr);
  __attribute__((opencl_global)) char *ptrAnnotated = nullptr;
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
#endif
  std::ignore = ptr;
  std::ignore = bytes;
  std::ignore = properties;
}
} // namespace detail

template <typename Properties = empty_properties_t>
void prefetch(void *ptr, Properties properties = {}) {
  detail::prefetch_impl(ptr, 1, properties);
}

template <typename Properties = empty_properties_t>
void prefetch(void *ptr, size_t bytes, Properties properties = {}) {
  detail::prefetch_impl(ptr, bytes, properties);
}

template <typename T, typename Properties = empty_properties_t>
void prefetch(T *ptr, Properties properties = {}) {
  prefetch((void *)ptr, sizeof(T), properties);
}

template <typename T, typename Properties = empty_properties_t>
void prefetch(T *ptr, size_t count, Properties properties = {}) {
  prefetch((void *)ptr, count * sizeof(T), properties);
}

// Only available if AddressSpace == global_space || AddressSpace ==
// generic_space
template <access::address_space AddressSpace, access::decorated IsDecorated,
          typename Properties = empty_properties_t>
std::enable_if_t<AddressSpace == access::address_space::global_space ||
                     AddressSpace == access::address_space::generic_space,
                 void>
prefetch(multi_ptr<void, AddressSpace, IsDecorated> ptr,
         Properties properties = {}) {
  prefetch(ptr.get(), properties);
}

// Only available if AddressSpace == global_space || AddressSpace ==
// generic_space
template <access::address_space AddressSpace, access::decorated IsDecorated,
          typename Properties = empty_properties_t>
std::enable_if_t<AddressSpace == access::address_space::global_space ||
                     AddressSpace == access::address_space::generic_space,
                 void>
prefetch(multi_ptr<void, AddressSpace, IsDecorated> ptr, size_t bytes,
         Properties properties = {}) {
  prefetch(ptr.get(), bytes, properties);
}

// Only available if AddressSpace == global_space || AddressSpace ==
// generic_space
template <typename T, access::address_space AddressSpace,
          access::decorated IsDecorated,
          typename Properties = empty_properties_t>
std::enable_if_t<AddressSpace == access::address_space::global_space ||
                     AddressSpace == access::address_space::generic_space,
                 void>
prefetch(multi_ptr<T, AddressSpace, IsDecorated> ptr,
         Properties properties = {}) {
  prefetch(ptr.get(), properties);
}

// Only available if AddressSpace == global_space || AddressSpace ==
// generic_space
template <typename T, access::address_space AddressSpace,
          access::decorated IsDecorated,
          typename Properties = empty_properties_t>
std::enable_if_t<AddressSpace == access::address_space::global_space ||
                     AddressSpace == access::address_space::generic_space,
                 void>
prefetch(multi_ptr<T, AddressSpace, IsDecorated> ptr, size_t count,
         Properties properties = {}) {
  prefetch(ptr.get(), count, properties);
}

// Only available if Dimensions > 0 && (AccessMode == read || AccessMode ==
// read_write)
template <typename DataT, int Dimensions, access_mode AccessMode,
          access::placeholder IsPlaceholder,
          typename Properties = empty_properties_t>
std::enable_if_t<(Dimensions > 0) && (AccessMode == access_mode::read ||
                                      AccessMode == access_mode::read_write),
                 void>
prefetch(
    accessor<DataT, Dimensions, AccessMode, target::device, IsPlaceholder> acc,
    id<Dimensions> offset, Properties properties = {}) {
  prefetch((void *)&acc[offset], sizeof(DataT), properties);
}

// Only available if Dimensions > 0 && (AccessMode == read || AccessMode ==
// read_write)
template <typename DataT, int Dimensions, access_mode AccessMode,
          access::placeholder IsPlaceholder,
          typename Properties = empty_properties_t>
std::enable_if_t<(Dimensions > 0) && (AccessMode == access_mode::read ||
                                      AccessMode == access_mode::read_write),
                 void>
prefetch(
    accessor<DataT, Dimensions, AccessMode, target::device, IsPlaceholder> acc,
    size_t offset, size_t count, Properties properties = {}) {
  prefetch((void *)&acc[offset], count * sizeof(DataT), properties);
}

template <typename Group, typename Properties = empty_properties_t>
typename std::enable_if_t<sycl::is_group_v<std::decay_t<Group>>, void>
joint_prefetch(Group g, void *ptr, Properties properties = {}) {
  detail::prefetch_impl(ptr, 1, properties);
}

template <typename Group, typename Properties = empty_properties_t>
typename std::enable_if_t<sycl::is_group_v<std::decay_t<Group>>, void>
joint_prefetch(Group g, void *ptr, size_t bytes, Properties properties = {}) {
  detail::prefetch_impl(ptr, bytes, properties);
}

template <typename Group, typename T, typename Properties = empty_properties_t>
typename std::enable_if_t<sycl::is_group_v<std::decay_t<Group>>, void>
joint_prefetch(Group g, T *ptr, Properties properties = {}) {
  joint_prefetch((void *)ptr, sizeof(T), properties);
}

template <typename Group, typename T, typename Properties = empty_properties_t>
typename std::enable_if_t<sycl::is_group_v<std::decay_t<Group>>, void>
joint_prefetch(Group g, T *ptr, size_t count, Properties properties = {}) {
  joint_prefetch((void *)ptr, count * sizeof(T), properties);
}

// Only available if AddressSpace == global_space || AddressSpace ==
// generic_space
template <typename Group, access::address_space AddressSpace,
          access::decorated IsDecorated,
          typename Properties = ext::oneapi::experimental::empty_properties_t>
typename std::enable_if_t<
    sycl::is_group_v<std::decay_t<Group>> &&
        (AddressSpace == access::address_space::global_space ||
         AddressSpace == access::address_space::generic_space),
    void>
joint_prefetch(Group g, multi_ptr<void, AddressSpace, IsDecorated> ptr,
               Properties properties = {}) {
  joint_prefetch(g, ptr.get(), properties);
}

// Only available if AddressSpace == global_space || AddressSpace ==
// generic_space
template <typename Group, access::address_space AddressSpace,
          access::decorated IsDecorated,
          typename Properties = ext::oneapi::experimental::empty_properties_t>
typename std::enable_if_t<
    sycl::is_group_v<std::decay_t<Group>> &&
        (AddressSpace == access::address_space::global_space ||
         AddressSpace == access::address_space::generic_space),
    void>
joint_prefetch(Group g, multi_ptr<void, AddressSpace, IsDecorated> ptr,
               size_t bytes, Properties properties = {}) {
  joint_prefetch(g, ptr.get(), bytes, properties);
}

// Only available if AddressSpace == global_space || AddressSpace ==
// generic_space
template <typename Group, typename T, access::address_space AddressSpace,
          access::decorated IsDecorated,
          typename Properties = ext::oneapi::experimental::empty_properties_t>
typename std::enable_if_t<
    sycl::is_group_v<std::decay_t<Group>> &&
        (AddressSpace == access::address_space::global_space ||
         AddressSpace == access::address_space::generic_space),
    void>
joint_prefetch(Group g, multi_ptr<T, AddressSpace, IsDecorated> ptr,
               Properties properties = {}) {
  joint_prefetch(g, ptr.get(), properties);
}

// Only available if AddressSpace == global_space || AddressSpace ==
// generic_space
template <typename Group, typename T, access::address_space AddressSpace,
          access::decorated IsDecorated,
          typename Properties = ext::oneapi::experimental::empty_properties_t>
typename std::enable_if_t<
    sycl::is_group_v<std::decay_t<Group>> &&
        (AddressSpace == access::address_space::global_space ||
         AddressSpace == access::address_space::generic_space),
    void>
joint_prefetch(Group g, multi_ptr<T, AddressSpace, IsDecorated> ptr,
               size_t count, Properties properties = {}) {
  joint_prefetch(g, ptr.get(), count, properties);
}

// Only available if Dimensions > 0 && (AccessMode == read || AccessMode ==
// read_write)
template <typename Group, typename DataT, int Dimensions,
          access_mode AccessMode, access::placeholder IsPlaceholder,
          typename Properties = ext::oneapi::experimental::empty_properties_t>
typename std::enable_if_t<sycl::is_group_v<std::decay_t<Group>> &&
                              (Dimensions > 0) &&
                              (AccessMode == access_mode::read ||
                               AccessMode == access_mode::read_write),
                          void>
joint_prefetch(
    Group g,
    accessor<DataT, Dimensions, AccessMode, target::device, IsPlaceholder> acc,
    size_t offset, Properties properties = {}) {
  joint_prefetch(g, (void *)&acc[offset], sizeof(DataT), properties);
}

// Only available if Dimensions > 0 && (AccessMode == read || AccessMode ==
// read_write)
template <typename Group, typename DataT, int Dimensions,
          access_mode AccessMode, access::placeholder IsPlaceholder,
          typename Properties = ext::oneapi::experimental::empty_properties_t>
typename std::enable_if_t<sycl::is_group_v<std::decay_t<Group>> &&
                              (Dimensions > 0) &&
                              (AccessMode == access_mode::read ||
                               AccessMode == access_mode::read_write),
                          void>
joint_prefetch(
    Group g,
    accessor<DataT, Dimensions, AccessMode, target::device, IsPlaceholder> acc,
    size_t offset, size_t count, Properties properties = {}) {
  joint_prefetch(g, (void *)&acc[offset], count * sizeof(DataT), properties);
}

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
