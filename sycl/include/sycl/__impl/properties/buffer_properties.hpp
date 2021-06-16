//==----------- buffer_properties.hpp --- SYCL buffer properties -----------==//
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
namespace buffer {
class use_host_ptr : public detail::DataLessProperty<detail::BufferUseHostPtr> {
};

class use_mutex : public detail::PropertyWithData<detail::BufferUseMutex> {
public:
  use_mutex(__sycl_internal::__v1::mutex_class &MutexRef) : MMutex(MutexRef) {}

  __sycl_internal::__v1::mutex_class *get_mutex_ptr() const { return &MMutex; }

private:
  __sycl_internal::__v1::mutex_class &MMutex;
};

class context_bound
    : public detail::PropertyWithData<detail::BufferContextBound> {
public:
  context_bound(__sycl_internal::__v1::context BoundContext) : MCtx(std::move(BoundContext)) {}

  __sycl_internal::__v1::context get_context() const { return MCtx; }

private:
  __sycl_internal::__v1::context MCtx;
};

class mem_channel : public detail::PropertyWithData<
                        detail::PropWithDataKind::BufferMemChannel> {
public:
  mem_channel(uint32_t Channel) : MChannel(Channel) {}
  uint32_t get_channel() const { return MChannel; }

private:
  uint32_t MChannel;
};

} // namespace buffer
} // namespace property

namespace ext {
namespace oneapi {
namespace property {
namespace buffer {

class use_pinned_host_memory
    : public detail::DataLessProperty<detail::BufferUsePinnedHostMemory> {};
} // namespace buffer
} // namespace property
} // namespace oneapi
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
