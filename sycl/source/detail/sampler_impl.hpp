//==----------------- sampler_impl.hpp - SYCL standard header file ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/__spirv/spirv_types.hpp>
#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/export.hpp>
#include <CL/sycl/property_list.hpp>

#include <unordered_map>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

enum class addressing_mode : unsigned int;
enum class filtering_mode : unsigned int;
enum class coordinate_normalization_mode : unsigned int;

namespace detail {
class __SYCL_EXPORT sampler_impl {
public:
  sampler_impl(coordinate_normalization_mode normalizationMode,
               addressing_mode addressingMode, filtering_mode filteringMode,
               const property_list &propList);

  sampler_impl(cl_sampler clSampler, const context &syclContext);

  addressing_mode get_addressing_mode() const;

  filtering_mode get_filtering_mode() const;

  coordinate_normalization_mode get_coordinate_normalization_mode() const;

  RT::PiSampler getOrCreateSampler(const context &Context);

  /// Checks if this sampler_impl has a property of type propertyT.
  ///
  /// \return true if this sampler_impl has a property of type propertyT.
  template <typename propertyT> bool has_property() const {
    return MPropList.has_property<propertyT>();
  }

  /// Gets the specified property of this sampler_impl.
  ///
  /// Throws invalid_object_error if this sampler_impl does not have a property
  /// of type propertyT.
  ///
  /// \return a copy of the property of type propertyT.
  template <typename propertyT> propertyT get_property() const {
    return MPropList.get_property<propertyT>();
  }

  ~sampler_impl();

private:
  /// Protects all the fields that can be changed by class' methods.
  std::mutex MMutex;

  std::unordered_map<context, RT::PiSampler> MContextToSampler;

  coordinate_normalization_mode MCoordNormMode;
  addressing_mode MAddrMode;
  filtering_mode MFiltMode;
  property_list MPropList;
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
