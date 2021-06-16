//==----------- image_properties.hpp --- SYCL image properties -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/__impl/context.hpp>
#include <sycl/__impl/detail/property_helper.hpp>

namespace __sycl_internal {
inline namespace __v1 {
namespace property {
namespace image {
class use_host_ptr : public detail::DataLessProperty<detail::ImageUseHostPtr> {
};

class use_mutex : public detail::PropertyWithData<detail::ImageUseMutex> {
public:
  use_mutex(__sycl_internal::__v1::mutex_class &MutexRef) : MMutex(MutexRef) {}

  __sycl_internal::__v1::mutex_class *get_mutex_ptr() const { return &MMutex; }

private:
  __sycl_internal::__v1::mutex_class &MMutex;
};

class context_bound
    : public detail::PropertyWithData<detail::ImageContextBound> {
public:
  context_bound(__sycl_internal::__v1::context BoundContext) : MCtx(std::move(BoundContext)) {}

  __sycl_internal::__v1::context get_context() const { return MCtx; }

private:
  __sycl_internal::__v1::context MCtx;
};
} // namespace image
} // namespace property
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
