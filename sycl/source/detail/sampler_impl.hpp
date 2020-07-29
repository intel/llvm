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

#include <unordered_map>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

enum class addressing_mode : unsigned int;
enum class filtering_mode : unsigned int;
enum class coordinate_normalization_mode : unsigned int;

namespace detail {
class __SYCL_EXPORT sampler_impl {
public:
  std::unordered_map<context, RT::PiSampler> m_contextToSampler;

private:
  coordinate_normalization_mode m_CoordNormMode;
  addressing_mode m_AddrMode;
  filtering_mode m_FiltMode;

public:
  sampler_impl(coordinate_normalization_mode normalizationMode,
               addressing_mode addressingMode, filtering_mode filteringMode);

  sampler_impl(cl_sampler clSampler, const context &syclContext);

  addressing_mode get_addressing_mode() const;

  filtering_mode get_filtering_mode() const;

  coordinate_normalization_mode get_coordinate_normalization_mode() const;

  RT::PiSampler getOrCreateSampler(const context &Context);

  ~sampler_impl();
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
