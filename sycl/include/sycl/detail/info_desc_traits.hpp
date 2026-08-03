//==-- info_desc_traits.hpp - SYCL info descriptor self-describing traits --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <unified-runtime/ur_api.h>

#include <type_traits>

// Self-describing info-descriptor model.
//
// A trait struct describes itself via these nested members:
//   * `using return_type = ...;`     return type of `get_info<Trait>()`
//   * `using info_class  = ...;`     class tag (one of the structs below)
//   * `static constexpr auto ur_code = ...;`  UR enum value
//
// `ur_code` is OPTIONAL. Traits dispatched via UR enum lookup must define it,
// of the family-specific UR enum type (see `ur_code_type` member of each
// info_class tag). Traits handled entirely SYCL-side simply omit `ur_code`;
// runtime dispatch hits an explicit `CASE` switch in the *_impl headers and
// the UR fallthrough is never instantiated for them.
//
// Most traits derive from `ur_traits_base` (UR-dispatched) or
// `rt_traits_base` (RT-only) and add `using return_type = ...;` in the
// derived body. The base intentionally does NOT carry `return_type` as a
// template parameter — that would mangle `std::string`/`std::list` into the
// derived layout name and trip the dual-ABI guard in
// `sycl/test/abi/sycl_classes_abi_neutral_test.cpp`.

namespace sycl {
inline namespace _V1 {
namespace detail {

namespace info_class {
// Common shape: each tag exposes the UR enum family used to query traits in
// this class. `UrInfoCode` static_asserts each trait's `ur_code` against this
// type so wrong-family enum values fail to compile with a clear diagnostic.
// Tags whose traits dispatch through multiple UR APIs (or none) leave
// `ur_code_type` as `void` to opt out of the family check.
template <typename T> struct info_class_base {
  using ur_code_type = T;
};

struct platform : info_class_base<ur_platform_info_t> {};
struct context : info_class_base<ur_context_info_t> {};
struct device : info_class_base<ur_device_info_t> {};
struct queue : info_class_base<ur_queue_info_t> {};
struct kernel : info_class_base<ur_kernel_info_t> {};
// kernel_device_specific traits dispatch through three different UR APIs
// (urKernelGetGroupInfo, urKernelGetSubGroupInfo, urKernelGetInfo) and so
// mix three native UR enum families. The runtime helper picks the right
// call via IsSubGroupInfo / IsKernelInfo trait tags, passing
// `UrInfoCode<T>::value` directly to the chosen API. We therefore allow
// each trait to declare ur_code with its own native UR enum type and skip
// the family static_assert by leaving ur_code_type as void.
struct kernel_device_specific : info_class_base<void> {};
// No UR enum lookup applies; queries dispatch through dedicated kernel/queue
// call paths. Kept for type uniformity.
struct kernel_queue_specific : info_class_base<void> {};
struct event : info_class_base<ur_event_info_t> {};
struct event_profiling : info_class_base<ur_profiling_info_t> {};
} // namespace info_class

// Detects that `T` is one of the `info_class::*` tags above. Holds when `T`
// inherits from any `info_class_base<U>` instantiation, so the check stays
// closed against future tags as long as they keep the same shape.
template <typename T, typename = void>
struct is_info_class_tag : std::false_type {};

template <typename T>
struct is_info_class_tag<T, std::void_t<typename T::ur_code_type>>
    : std::is_base_of<info_class::info_class_base<typename T::ur_code_type>,
                      T> {};

// Common base for UR-dispatched traits. Derived structs inherit `info_class`
// and `ur_code` and add `using return_type = ...;` in their body. `auto
// UrCode` lets each derived trait pass its own native UR enum type
// (ur_device_info_t, ur_kernel_group_info_t, etc.) without forcing a single
// enum family at the base. Note: `return_type` is intentionally NOT a base
// template parameter — see file-level comment.
template <typename ClassT, auto UrCode> struct ur_traits_base {
  static_assert(is_info_class_tag<ClassT>::value,
                "ur_traits_base ClassT must be one of the info_class::* tags "
                "(i.e. derive from info_class::info_class_base).");
  using info_class = ClassT;
  static constexpr decltype(UrCode) ur_code = UrCode;
};

// Common base for RT-only traits (no UR enum lookup). Derived structs inherit
// `info_class` and add `using return_type = ...;`. Runtime dispatch hits an
// explicit CASE in *_impl.hpp.
template <typename ClassT> struct rt_traits_base {
  static_assert(is_info_class_tag<ClassT>::value,
                "rt_traits_base ClassT must be one of the info_class::* tags "
                "(i.e. derive from info_class::info_class_base).");
  using info_class = ClassT;
};

// Detects whether T looks like a self-describing info trait, i.e. it carries
// both `return_type` and `info_class` members. Used to gate cross-checks that
// only make sense once both members are present (e.g. UR enum family
// validation in `UrInfoCode`).
template <typename T, typename = void>
struct is_self_describing_info_desc : std::false_type {};

template <typename T>
struct is_self_describing_info_desc<
    T, std::void_t<typename T::return_type, typename T::info_class>>
    : std::true_type {};

// Detects whether T carries a `ur_code` static member. RT-only traits omit
// `ur_code`; runtime code paths gate UR fallthrough on this trait so the
// `UrInfoCode<T>` template is never instantiated for non-UR traits.
template <typename T, typename = void>
struct is_ur_dispatched : std::false_type {};

template <typename T>
struct is_ur_dispatched<T, std::void_t<decltype(T::ur_code)>> : std::true_type {
};

// Per-class type predicate plus return-type accessor. Used by per-object
// `get_info_impl<T>()` dispatch helpers (e.g. `is_device_info_desc<T>`) to
// confine each object's `get_info` to traits in the matching info_class
// family, and to expose the trait's `return_type` for the function's return
// signature. The `return_type` typedef here is load-bearing for ABI symbol
// mangling — keep stable.
template <typename T, typename Class, typename = void>
struct is_info_desc_for : std::false_type {};

template <typename T, typename Class>
struct is_info_desc_for<
    T, Class,
    std::enable_if_t<is_self_describing_info_desc<T>::value &&
                     std::is_same_v<typename T::info_class, Class>>>
    : std::true_type {
  using return_type = typename T::return_type;
};

// `is_backend_info_desc<T>` is not info-class-tied; specializations belong to
// per-backend specs. Removing would break the spec-mandated
// `get_backend_info<>()` member, so the primary-template `false_type` lives
// alongside the SFINAE primitives until a backend ships descriptors. Same
// gcc/clang mangling workaround as the per-class `is_*_info_desc` predicates:
// using `T::return_type` instead of `std::enable_if<...>::type` avoids a
// missing `E` terminator in gcc's unresolved-qualifier-level sequence.
template <typename T> struct is_backend_info_desc : std::false_type {};

// kernel_queue_specific has no per-class `info` header (no UR enum lookup);
// queries dispatch through dedicated kernel/queue paths. Park the predicate
// alongside `is_backend_info_desc`.
template <typename T>
struct is_kernel_queue_specific_info_desc
    : is_info_desc_for<T, info_class::kernel_queue_specific> {};

} // namespace detail
} // namespace _V1
} // namespace sycl
