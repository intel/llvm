//==----------------- sampler_impl.hpp - SYCL standard header file ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/__spirv/spirv_types.hpp>
#include <sycl/detail/export.hpp>
#include <sycl/detail/ur.hpp>
#include <sycl/property_list.hpp>

#include <mutex>
#include <unordered_map>

#ifdef __SYCL_INTERNAL_API
#include <sycl/detail/cl.h>
#endif

namespace sycl {
inline namespace _V1 {

enum class addressing_mode : unsigned int;
enum class filtering_mode : unsigned int;
enum class coordinate_normalization_mode : unsigned int;

namespace detail {

class context_impl;
using ContextImplPtr = std::shared_ptr<context_impl>;

class sampler_impl {
public:
  sampler_impl(coordinate_normalization_mode normalizationMode,
               addressing_mode addressingMode, filtering_mode filteringMode,
               const property_list &propList);

  sampler_impl(cl_sampler clSampler, const ContextImplPtr &syclContext);

  addressing_mode get_addressing_mode() const;

  filtering_mode get_filtering_mode() const;

  coordinate_normalization_mode get_coordinate_normalization_mode() const;

  ur_sampler_handle_t getOrCreateSampler(const ContextImplPtr &ContextImpl);

  ~sampler_impl();

  const property_list &getPropList() const { return MPropList; }

private:
  /// Protects all the fields that can be changed by class' methods.
  std::mutex MMutex;

  std::unordered_map<ContextImplPtr, ur_sampler_handle_t> MContextToSampler;

  coordinate_normalization_mode MCoordNormMode;
  addressing_mode MAddrMode;
  filtering_mode MFiltMode;
  property_list MPropList;

  void verifyProps(const property_list &Props) const;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
