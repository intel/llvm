//==----------- buffer_properties.hpp --- SYCL buffer properties -----------==//
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
namespace buffer {
class use_host_ptr : public detail::DataLessProperty<detail::BufferUseHostPtr> {
};

class use_mutex : public detail::PropertyWithData<detail::BufferUseMutex> {
public:
  use_mutex(std::mutex &MutexRef) : MMutex(MutexRef) {}

  std::mutex *get_mutex_ptr() const { return &MMutex; }

private:
  std::mutex &MMutex;
};

class context_bound
    : public detail::PropertyWithData<detail::BufferContextBound> {
public:
  context_bound(sycl::context BoundContext) : MCtx(std::move(BoundContext)) {}

  sycl::context get_context() const { return MCtx; }

private:
  sycl::context MCtx;
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

class use_pinned_host_memory : public sycl::detail::DataLessProperty<
                                   sycl::detail::BufferUsePinnedHostMemory> {};
} // namespace buffer
} // namespace property
} // namespace oneapi
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
