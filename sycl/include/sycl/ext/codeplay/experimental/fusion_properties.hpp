//==----------- fusion_properties.hpp --- SYCL fusion properties -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/access/access.hpp>          // for mode, placeholder
#include <sycl/detail/property_helper.hpp> // for DataLessPropKind, Dat...
#include <sycl/ext/oneapi/experimental/common_annotated_properties/properties.hpp>
#include <sycl/properties/property_traits.hpp> // for is_property, is_prope...
#include <sycl/queue.hpp>                      // for queue

#include <type_traits> // for true_type

namespace sycl {
inline namespace _V1 {
// Kernel fusion properties
namespace ext::codeplay::experimental::property {

class promote_private
    : public detail::DataLessProperty<detail::FusionPromotePrivate> {};

class promote_local
    : public detail::DataLessProperty<detail::FusionPromoteLocal> {};

class no_barriers : public detail::DataLessProperty<detail::FusionNoBarrier> {};

class force_fusion : public detail::DataLessProperty<detail::FusionForce> {};

namespace queue {
class enable_fusion : public detail::DataLessProperty<detail::FusionEnable> {};
} // namespace queue

} // namespace ext::codeplay::experimental::property

// Graph fusion properties
namespace ext::oneapi::experimental {
namespace property {

// Helpers for template properties
namespace detail {

template <sycl::memory_scope Scope>
struct enable_if_valid_internalization_scope
    : std::enable_if<Scope == sycl::memory_scope::work_group ||
                     Scope == sycl::memory_scope::work_item> {};

template <sycl::memory_scope Scope>
using enable_if_valid_internalization_scope_t =
    typename enable_if_valid_internalization_scope<Scope>::type;

template <bool B, auto TKind, auto FKind> struct conditional_property_kind {
  static constexpr auto value = TKind;
};

template <auto TKind, auto FKind>
struct conditional_property_kind<false, TKind, FKind> {
  static constexpr auto value = FKind;
};

template <sycl::memory_scope Scope, auto WGKind, auto WIKind>
struct internalization_scope_conditional
    : conditional_property_kind<Scope == sycl::memory_scope::work_group, WGKind,
                                WIKind> {};

template <sycl::memory_scope Scope, auto WGKind, auto WIKind>
inline constexpr auto internalization_scope_conditional_v =
    internalization_scope_conditional<Scope, WGKind, WIKind>::value;

} // namespace detail

// 'access_scope<sycl::memory_scope>' and 'fusion_internal_memory' are runtime
// and compile-time properties at the same time. This is achieved by deriving
// these classes from 'DataLessProperty' (for runtime property) and
// 'property_value' (for compile-time property) and adding another class with
// '_key' suffix that inherits from 'compile_time_property_key' and uses the
// property class as 'value_t'.

template <sycl::memory_scope Scope,
          sycl::ext::oneapi::experimental::detail::PropKind PropKind>
struct access_scope_key;

// 'access_scope<sycl::memory_scope>' is a template property. Currently, only
// specializations with 'memory_scope::work_item' and 'memory_scope::work_group'
// are valid. Depending on the scope, different values for 'DataLessPropKind'
// (for runtime property) and 'PropKind' (for compile-time property) will be
// used.
template <
    sycl::memory_scope Scope,
    typename = detail::enable_if_valid_internalization_scope_t<Scope>,
    ::sycl::detail::DataLessPropKind PropID =
        detail::internalization_scope_conditional_v<
            Scope, ::sycl::detail::AccessScopeWorkGroup,
            ::sycl::detail::AccessScopeWorkItem>,
    sycl::ext::oneapi::experimental::detail::PropKind PropKind =
        detail::internalization_scope_conditional_v<
            Scope,
            sycl::ext::oneapi::experimental::detail::PropKind::AccessScopeWG,
            sycl::ext::oneapi::experimental::detail::PropKind::AccessScopeWI>>
struct access_scope : ::sycl::detail::DataLessProperty<PropID>,
                      property_value<access_scope_key<Scope, PropKind>> {};

template <
    sycl::memory_scope Scope,
    sycl::ext::oneapi::experimental::detail::PropKind PropKind =
        detail::internalization_scope_conditional_v<
            Scope,
            sycl::ext::oneapi::experimental::detail::PropKind::AccessScopeWG,
            sycl::ext::oneapi::experimental::detail::PropKind::AccessScopeWI>>
struct access_scope_key
    : sycl::ext::oneapi::experimental::detail::compile_time_property_key<
          PropKind> {
  using value_t = access_scope<Scope>;
};

struct fusion_internal_memory_key;

struct fusion_internal_memory
    : ::sycl::detail::DataLessProperty<::sycl::detail::FusionInternalMemory>,
      property_value<fusion_internal_memory_key> {};

struct fusion_internal_memory_key
    : ::sycl::ext::oneapi::experimental::detail::compile_time_property_key<
          ::sycl::ext::oneapi::experimental::detail::PropKind::
              FusionInternalMemory> {
  using value_t = fusion_internal_memory;
};

struct fusion_no_init_key
    : ::sycl::ext::oneapi::experimental::detail::compile_time_property_key<
          ::sycl::ext::oneapi::experimental::detail::PropKind::FusionNoInit> {
  using value_t = property_value<fusion_no_init_key>;
};

using fusion_no_init = fusion_no_init_key::value_t;

} // namespace property

inline constexpr property::access_scope<sycl::memory_scope::work_group>
    access_scope_work_group;

inline constexpr property::access_scope<sycl::memory_scope::work_item>
    access_scope_work_item;

inline constexpr property::fusion_internal_memory fusion_internal_memory;

inline constexpr property::fusion_no_init fusion_no_init;

// Property trait specializations for compile-time properties.
template <>
struct is_property_key<property::fusion_internal_memory_key> : std::true_type {
};

template <sycl::memory_scope Scope>
struct is_property_key<property::access_scope<Scope>> : std::true_type {};

template <>
struct is_property_key<property::fusion_no_init> : std::true_type {};

template <>
struct detail::IsCompileTimeProperty<property::fusion_internal_memory_key>
    : std::true_type {};

template <sycl::memory_scope Scope>
struct detail::IsCompileTimeProperty<property::access_scope<Scope>>
    : std::true_type {};

template <>
struct detail::IsCompileTimeProperty<property::fusion_no_init>
    : std::true_type {};

template <>
struct detail::IsCompileTimePropertyValue<property::fusion_internal_memory>
    : std::true_type {};

template <sycl::memory_scope Scope>
struct detail::IsCompileTimePropertyValue<property::access_scope<Scope>>
    : std::true_type {};

template <>
struct detail::PropertyID<property::fusion_internal_memory>
    : detail::PropertyID<property::fusion_internal_memory_key> {};

template <sycl::memory_scope Scope>
struct detail::PropertyID<property::access_scope<Scope>>
    : detail::PropertyID<property::access_scope_key<Scope>> {};

// Forward declaration of annotated_ptr
template <typename T, typename PropertyListT> class annotated_ptr;

// Property trait specializations for annotated_ptr
template <typename T, typename PropertyListT>
struct is_property_key_of<property::fusion_internal_memory_key,
                          annotated_ptr<T, PropertyListT>> : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<property::access_scope<sycl::memory_scope::work_item>,
                          annotated_ptr<T, PropertyListT>> : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<
    property::access_scope<sycl::memory_scope::work_group>,
    annotated_ptr<T, PropertyListT>> : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<property::fusion_no_init,
                          annotated_ptr<T, PropertyListT>> : std::true_type {};

template <typename T>
struct is_valid_property<T, property::fusion_internal_memory>
    : std::bool_constant<std::is_pointer_v<T>> {};
template <typename T>
struct is_valid_property<T,
                         property::access_scope<sycl::memory_scope::work_item>>
    : std::bool_constant<std::is_pointer_v<T>> {};
template <typename T>
struct is_valid_property<T,
                         property::access_scope<sycl::memory_scope::work_group>>
    : std::bool_constant<std::is_pointer_v<T>> {};
template <typename T>
struct is_valid_property<T, property::fusion_no_init>
    : std::bool_constant<std::is_pointer_v<T>> {};

// Trait specializations to make compile-time properties available to the device
// compiler.
namespace detail {
template <> struct PropertyMetaInfo<property::fusion_internal_memory> {
  static constexpr const char *name = "sycl-fusion-internal-mem";
  static constexpr int value = 1;
};
template <>
struct PropertyMetaInfo<property::access_scope<sycl::memory_scope::work_item>> {
  static constexpr const char *name = "sycl-access-scope-work-item";
  static constexpr int value = 1;
};
template <>
struct PropertyMetaInfo<
    property::access_scope<sycl::memory_scope::work_group>> {
  static constexpr const char *name = "sycl-access-scope-work-group";
  static constexpr int value = 1;
};
template <> struct PropertyMetaInfo<property::fusion_no_init> {
  static constexpr const char *name = "sycl-fusion-no-init";
  static constexpr int value = 1;
};
} // namespace detail
} // namespace ext::oneapi::experimental

// Forward declarations
template <typename T, int Dimensions, typename AllocatorT, typename Enable>
class buffer;

template <typename DataT, int Dimensions, access::mode AccessMode,
          access::target AccessTarget, access::placeholder IsPlaceholder,
          typename PropertyListT>
class accessor;

class queue;

// Property trait specializations.
template <>
struct is_property<ext::codeplay::experimental::property::promote_private>
    : std::true_type {};

template <>
struct is_property<ext::codeplay::experimental::property::promote_local>
    : std::true_type {};

template <>
struct is_property<ext::oneapi::experimental::property::fusion_internal_memory>
    : std::true_type {};

template <sycl::memory_scope Scope>
struct is_property<ext::oneapi::experimental::property::access_scope<Scope>>
    : std::true_type {};

template <>
struct is_property<ext::codeplay::experimental::property::no_barriers>
    : std::true_type {};

template <>
struct is_property<ext::codeplay::experimental::property::force_fusion>
    : std::true_type {};

template <>
struct is_property<ext::codeplay::experimental::property::queue::enable_fusion>
    : std::true_type {};

// Buffer property trait specializations
template <typename T, int Dimensions, typename AllocatorT>
struct is_property_of<ext::codeplay::experimental::property::promote_private,
                      buffer<T, Dimensions, AllocatorT, void>>
    : std::true_type {};

template <typename T, int Dimensions, typename AllocatorT>
struct is_property_of<ext::codeplay::experimental::property::promote_local,
                      buffer<T, Dimensions, AllocatorT, void>>
    : std::true_type {};

template <typename T, int Dimensions, typename AllocatorT>
struct is_property_of<
    ext::oneapi::experimental::property::fusion_internal_memory,
    buffer<T, Dimensions, AllocatorT, void>> : std::true_type {};

template <sycl::memory_scope Scope, typename T, int Dimensions,
          typename AllocatorT>
struct is_property_of<ext::oneapi::experimental::property::access_scope<Scope>,
                      buffer<T, Dimensions, AllocatorT, void>>
    : std::true_type {};

// Accessor property trait specializations
template <typename DataT, int Dimensions, access::mode AccessMode,
          access::target AccessTarget, access::placeholder IsPlaceholder,
          typename PropertyListT>
struct is_property_of<ext::codeplay::experimental::property::promote_private,
                      accessor<DataT, Dimensions, AccessMode, AccessTarget,
                               IsPlaceholder, PropertyListT>> : std::true_type {
};

template <typename DataT, int Dimensions, access::mode AccessMode,
          access::target AccessTarget, access::placeholder IsPlaceholder,
          typename PropertyListT>
struct is_property_of<ext::codeplay::experimental::property::promote_local,
                      accessor<DataT, Dimensions, AccessMode, AccessTarget,
                               IsPlaceholder, PropertyListT>> : std::true_type {
};

template <typename DataT, int Dimensions, access::mode AccessMode,
          access::target AccessTarget, access::placeholder IsPlaceholder,
          typename PropertyListT>
struct is_property_of<
    ext::oneapi::experimental::property::fusion_internal_memory,
    accessor<DataT, Dimensions, AccessMode, AccessTarget, IsPlaceholder,
             PropertyListT>> : std::true_type {};

template <sycl::memory_scope Scope, typename DataT, int Dimensions,
          access::mode AccessMode, access::target AccessTarget,
          access::placeholder IsPlaceholder, typename PropertyListT>
struct is_property_of<ext::oneapi::experimental::property::access_scope<Scope>,
                      accessor<DataT, Dimensions, AccessMode, AccessTarget,
                               IsPlaceholder, PropertyListT>> : std::true_type {
};

// Queue property trait specializations
template <>
struct is_property_of<
    ext::codeplay::experimental::property::queue::enable_fusion, queue>
    : std::true_type {};

} // namespace _V1
} // namespace sycl
