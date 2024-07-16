//==----------------- sampler.hpp - SYCL standard header file --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/access/access.hpp>     // for mode, placeholder, target
#include <sycl/detail/defines.hpp>    // for __SYCL_SPECIAL_CLASS, __SYCL_TYPE
#include <sycl/detail/export.hpp>     // for __SYCL_EXPORT
#include <sycl/detail/impl_utils.hpp> // for getSyclObjImpl
#include <sycl/detail/pi.h>           // for PI_SAMPLER_ADDRESSING_MODE_CLAMP
#include <sycl/property_list.hpp>     // for property_list

#include <cstddef> // for size_t
#include <memory>  // for shared_ptr, hash
#include <variant> // for hash

namespace sycl {
inline namespace _V1 {
enum class addressing_mode : unsigned int {
  mirrored_repeat = PI_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT,
  repeat = PI_SAMPLER_ADDRESSING_MODE_REPEAT,
  clamp_to_edge = PI_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE,
  clamp = PI_SAMPLER_ADDRESSING_MODE_CLAMP,
  none = PI_SAMPLER_ADDRESSING_MODE_NONE
};

enum class filtering_mode : unsigned int {
  nearest = PI_SAMPLER_FILTER_MODE_NEAREST,
  linear = PI_SAMPLER_FILTER_MODE_LINEAR
};

enum class coordinate_normalization_mode : unsigned int {
  normalized = 1,
  unnormalized = 0
};

namespace detail {
template <typename DataT, int Dimensions, access::mode AccessMode,
          access::target AccessTarget, access::placeholder IsPlaceholder>
class image_accessor;
}

namespace detail {
#ifdef __SYCL_DEVICE_ONLY__
class __SYCL_EXPORT sampler_impl {
public:
  sampler_impl() = default;

  sampler_impl(__ocl_sampler_t Sampler) : m_Sampler(Sampler) {}

  ~sampler_impl() = default;

  __ocl_sampler_t m_Sampler;
};
#else
class sampler_impl;
#endif
} // namespace detail

/// Encapsulates a configuration for sampling an image accessor.
///
/// \sa sycl_api_acc
///
/// \ingroup sycl_api
class __SYCL_EXPORT __SYCL_SPECIAL_CLASS __SYCL_TYPE(sampler) sampler {
public:
  sampler(coordinate_normalization_mode normalizationMode,
          addressing_mode addressingMode, filtering_mode filteringMode,
          const property_list &propList = {});

#ifdef __SYCL_INTERNAL_API
  sampler(cl_sampler clSampler, const context &syclContext);
#endif

  sampler(const sampler &rhs) = default;

  sampler(sampler &&rhs) = default;

  sampler &operator=(const sampler &rhs) = default;

  sampler &operator=(sampler &&rhs) = default;

  bool operator==(const sampler &rhs) const;

  bool operator!=(const sampler &rhs) const;

  /// Checks if this sampler has a property of type propertyT.
  ///
  /// \return true if this sampler has a property of type propertyT.
  template <typename propertyT> bool has_property() const noexcept {
    return getPropList().template has_property<propertyT>();
  }

  /// Gets the specified property of this sampler.
  ///
  /// Throws an exception with errc::invalid if this sampler does not have a
  /// property of type propertyT.
  ///
  /// \return a copy of the property of type propertyT.
  template <typename propertyT> propertyT get_property() const {
    return getPropList().template get_property<propertyT>();
  }

  addressing_mode get_addressing_mode() const;

  filtering_mode get_filtering_mode() const;

  coordinate_normalization_mode get_coordinate_normalization_mode() const;

private:
#ifdef __SYCL_DEVICE_ONLY__
  detail::sampler_impl impl;
  void __init(__ocl_sampler_t Sampler) { impl.m_Sampler = Sampler; }
  char padding[sizeof(std::shared_ptr<detail::sampler_impl>) - sizeof(impl)];

public:
  sampler() = default;

private:
#else
  std::shared_ptr<detail::sampler_impl> impl;
  template <class Obj>
  friend const decltype(Obj::impl) &
  detail::getSyclObjImpl(const Obj &SyclObject);
#endif
  template <typename DataT, int Dimensions, sycl::access::mode AccessMode,
            sycl::access::target AccessTarget,
            access::placeholder IsPlaceholder>
  friend class detail::image_accessor;

  const property_list &getPropList() const;
};

// SYCL 2020 image_sampler struct
struct image_sampler {
  addressing_mode addressing;
  coordinate_normalization_mode coordinate;
  filtering_mode filtering;
};

} // namespace _V1
} // namespace sycl

namespace std {
template <> struct hash<sycl::sampler> {
  size_t operator()(const sycl::sampler &s) const {
#ifdef __SYCL_DEVICE_ONLY__
    (void)s;
    return 0;
#else
    return hash<std::shared_ptr<sycl::detail::sampler_impl>>()(
        sycl::detail::getSyclObjImpl(s));
#endif
  }
};
} // namespace std
