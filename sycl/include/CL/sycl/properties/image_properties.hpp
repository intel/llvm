//==----------- image_properties.hpp --- SYCL image properties -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/property_helper.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace property {
namespace image {
class use_host_ptr : public detail::DataLessProperty<detail::ImageUseHostPtr> {
};

class use_mutex : public detail::PropertyWithData<detail::ImageUseMutex> {
public:
  use_mutex(std::mutex &MutexRef) : MMutex(MutexRef) {}

  std::mutex *get_mutex_ptr() const { return &MMutex; }

private:
  std::mutex &MMutex;
};

class context_bound
    : public detail::PropertyWithData<detail::ImageContextBound> {
public:
  context_bound(sycl::context BoundContext) : MCtx(std::move(BoundContext)) {}

  sycl::context get_context() const { return MCtx; }

private:
  sycl::context MCtx;
};
} // namespace image
} // namespace property
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
