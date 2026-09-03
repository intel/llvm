//==---------- property.hpp --- SYCL compile-time property engine ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared, namespace-neutral core of the "new-style" compile-time property
// engine: the `PropKind` registry and the property key/value base classes that
// property definitions are built on. This lives in `sycl::detail` so that both
// the `sycl::ext::oneapi::experimental` extensions and the `sycl::khr` KHR
// properties layer can define their properties on top of a single registry.
//
// NOTE: This header deliberately contains only the property-*definition*
// infrastructure -- the pieces that appear inside a property type's definition
// but never in a public API's mangled signature. The property list container
// (`properties`), `property_value`, the list traits (`is_property_list`,
// `is_property_value`), and the applicability/customization traits
// (`is_property_key_of`, `PropertyMetaInfo`, ...) remain in the facades
// (see <sycl/ext/oneapi/properties/property.hpp>) so that moving this engine
// does not change any mangled symbol name. See the "HOW-TO" comments in that
// facade header for how to define a new property.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <stdint.h>    // for uint32_t
#include <type_traits> // for is_same_v

#include <sycl/detail/defines_elementary.hpp>

namespace sycl {
inline namespace _V1 {
// Forward declaration of the experimental property list container so the
// property key/value base classes can befriend it across namespaces (the
// container accesses the protected `get_property_impl`).
namespace ext::oneapi::experimental {
template <typename> class __SYCL_EBO properties;
} // namespace ext::oneapi::experimental

namespace detail {

// List of all properties.
//
// This is a scoped enumeration so that its enumerators do not leak into the
// enclosing `sycl::detail` namespace, where they would otherwise collide with
// the enumerators of the old-style property enums (DataLessPropKind /
// PropWithDataKind in <sycl/detail/property_helper.hpp>).
enum class PropKind : uint32_t {
  DeviceImageScope = 0,
  HostAccess = 1,
  WorkGroupSize = 2,
  WorkGroupSizeHint = 3,
  SubGroupSize = 4,
  DeviceHas = 5,
  Alignment = 6,
  CacheConfig = 7,
  UseRootSync = 8,
  GRFSize = 9,
  GRFSizeAutomatic = 10,
  ESIMDL1CacheHint = 11,
  ESIMDL2CacheHint = 12,
  ESIMDL3CacheHint = 13,
  UsmKind = 14,
  CacheControlReadHint = 15,
  CacheControlReadAssertion = 16,
  CacheControlWrite = 17,
  BuildOptions = 18,
  BuildLog = 19,
  FloatingPointControls = 20,
  DataPlacement = 21,
  ContiguousMemory = 22,
  FullGroup = 23,
  Naive = 24,
  WorkGroupProgress = 25,
  SubGroupProgress = 26,
  WorkItemProgress = 27,
  NDRangeKernel = 28,
  SingleTaskKernel = 29,
  IndirectlyCallable = 30,
  CallsIndirectly = 31,
  InputDataPlacement = 32,
  OutputDataPlacement = 33,
  IncludeFiles = 34,
  RegisteredNames = 35,
  ClusterLaunch = 36,
  MaxWorkGroupSize = 37,
  MaxLinearWorkGroupSize = 38,
  Prefetch = 39,
  Deterministic = 40,
  InitializeToIdentity = 41,
  WorkGroupScratchSize = 42,
  Unaliased = 43,
  EventMode = 44,
  NativeLocalBlockIO = 45,
  InitialThreshold = 46,
  MaximumSize = 47,
  ZeroInit = 48,
  FastLink = 49,
  EnableIPC = 50,
  RegisterHostMemoryReadOnly = 51,
  EnableProfiling = 52,
  MaximumRegisters = 53,
  MaximumRegistersAutomatic = 54,
  // PropKindSize must always be the last value.
  PropKindSize = 55,
};

template <typename PropertyT> struct PropertyToKind {
  static constexpr PropKind Kind = PropertyT::Kind;
};

struct property_tag {};

// This is used to implement has/get_property via inheritance queries.
template <typename property_key_t> struct property_key_tag : property_tag {};

template <typename property_t, PropKind Kind,
          typename property_key_t = property_t>
struct property_base : property_key_tag<property_key_t> {
  using key_t = property_key_t;

protected:
  constexpr property_t get_property_impl(property_key_tag<key_t>) const {
    return *static_cast<const property_t *>(this);
  }

  // For key_t access in error reporting specialization. The property list
  // container lives in the experimental facade namespace.
  template <typename>
  friend class __SYCL_EBO ext::oneapi::experimental::properties;

#if !defined(_MSC_VER)
  // Temporary, to ensure new code matches previous behavior and to catch any
  // silly copy-paste mistakes. MSVC can't compile it, but linux-only is
  // enough for this temporary check.
  static_assert([]() constexpr -> bool {
    if constexpr (std::is_same_v<property_t, key_t>)
      // key_t is incomplete at this point for runtime properties.
      return true;
    else
      return Kind == PropertyToKind<key_t>::Kind;
  }());
#endif
};

struct property_key_base_tag {};
struct compile_time_property_key_base_tag : property_key_base_tag {};

template <typename property_t, PropKind Kind_>
struct run_time_property_key : property_key_base_tag,
                               property_base<property_t, Kind_> {
protected:
  static constexpr PropKind Kind = Kind_;

  template <typename T> friend struct PropertyToKind;
};

template <PropKind Kind_>
struct compile_time_property_key : compile_time_property_key_base_tag {
protected:
  static constexpr PropKind Kind = Kind_;

  template <typename T> friend struct PropertyToKind;
};

// Get unique ID for property.
template <typename PropertyT> struct PropertyID {
  static constexpr int value =
      static_cast<int>(PropertyToKind<PropertyT>::Kind);
};

} // namespace detail
} // namespace _V1
} // namespace sycl
