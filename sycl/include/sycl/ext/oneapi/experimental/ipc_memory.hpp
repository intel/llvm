//==------- ipc_memory.hpp --- SYCL inter-process communicable memory ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp>
#include <sycl/detail/export.hpp>
#include <sycl/detail/owner_less_base.hpp>
#include <sycl/sycl_span.hpp>

#include <memory>

namespace sycl {
inline namespace _V1 {

class context;
class device;

namespace detail {
class ipc_memory_impl;
}

namespace ext::oneapi::experimental {
using ipc_memory_handle_data_t = span<char, sycl::dynamic_extent>;

class __SYCL_EXPORT ipc_memory
    : public sycl::detail::OwnerLessBase<ipc_memory> {
public:
  ipc_memory(void *Ptr, const sycl::context &Ctx);

  void put();

  static void *open(ipc_memory_handle_data_t IPCMemoryHandleData,
                    const sycl::context &Ctx, const sycl::device &Dev);
  static void close(void *Ptr, const sycl::context &Ctx);

  ipc_memory_handle_data_t get_handle_data() const;

  void *get_ptr() const;

private:
  ipc_memory(std::shared_ptr<sycl::detail::ipc_memory_impl> IPCMemImpl)
      : impl{IPCMemImpl} {}

  std::shared_ptr<sycl::detail::ipc_memory_impl> impl;

  template <class Obj>
  friend const decltype(Obj::impl) &
  sycl::detail::getSyclObjImpl(const Obj &SyclObject);

  template <class T>
  friend T sycl::detail::createSyclObjFromImpl(
      std::add_rvalue_reference_t<decltype(T::impl)> ImplObj);
  template <class T>
  friend T sycl::detail::createSyclObjFromImpl(
      std::add_lvalue_reference_t<const decltype(T::impl)> ImplObj);
};
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
