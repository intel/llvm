//==--------- buffer_properties.hpp --- SYCL buffer properties -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/property_base.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace property {
namespace buffer {

class use_host_ptr : public detail::property_base {};

class use_mutex : public detail::property_base {
public:
  use_mutex(mutex_class &MutexRef) : m_MutexClass(MutexRef) {}
  mutex_class *get_mutex_ptr() const { return &m_MutexClass; }

private:
  mutex_class &m_MutexClass;
};

class context_bound : public detail::property_base {
public:
  context_bound(context Context) : m_Context(Context) {}
  context get_context() const { return m_Context; }

private:
  cl::sycl::context m_Context;
};

} // namespace buffer
} // namespace property
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
