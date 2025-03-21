//===- usm_props.hpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/feature_test.hpp>
#include <sycl/properties/property_traits.hpp> // for is_property

#include <type_traits> // for true_type

namespace sycl {
inline namespace _V1 {
// There's a lot of dull-as-ditchwater C++ boilerplate here, so let's use a
// macro, eh?
#define __DEF_PROP(EXT_NS, PROP_NAME, ENUMERATOR)                              \
  namespace ext {                                                              \
  namespace EXT_NS {                                                           \
  class PROP_NAME                                                              \
      : public sycl::detail::DataLessProperty<sycl::detail::ENUMERATOR> {      \
  public:                                                                      \
    PROP_NAME() = default;                                                     \
  };                                                                           \
  }                                                                            \
  }                                                                            \
  template <> struct is_property<ext::EXT_NS::PROP_NAME> : std::true_type {};

__DEF_PROP(codeplay::usm_props, host_hot, HostHot)
__DEF_PROP(codeplay::usm_props, device_hot, DeviceHot)
__DEF_PROP(codeplay::usm_props, host_cold, HostCold)
__DEF_PROP(codeplay::usm_props, device_cold, DeviceCold)
__DEF_PROP(codeplay::usm_props, host_cache_non_coherent, HostCacheNonCoherent)
__DEF_PROP(codeplay::usm_props, device_cache_non_coherent,
           DeviceCacheNonCoherent)
__DEF_PROP(codeplay::usm_props, host_cache_write_combine, HostCacheWriteCombine)
__DEF_PROP(codeplay::usm_props, device_cache_write_combine,
           DeviceCacheWriteCombine)
__DEF_PROP(codeplay::usm_props, host_access_sequential, HostAccessSequential)
__DEF_PROP(codeplay::usm_props, device_access_sequential,
           DeviceAccessSequential)
__DEF_PROP(codeplay::usm_props, host_access_random, HostAccessRandom)
__DEF_PROP(codeplay::usm_props, device_access_random, DeviceAccessRandom)
__DEF_PROP(codeplay::usm_props, host_read_only, HostReadOnly)
__DEF_PROP(codeplay::usm_props, device_read_only, DeviceReadOnly)
__DEF_PROP(codeplay::usm_props, host_write_only, HostWriteOnly)
__DEF_PROP(codeplay::usm_props, device_write_only, DeviceWriteOnly)

#undef __DEF_PROP

} // namespace _V1
} // namespace sycl
