//==--- physical_mem.hpp - sycl_ext_oneapi_virtual_mem physical_mem class --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/access/access.hpp>
#include <sycl/context.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/owner_less_base.hpp>
#include <sycl/device.hpp>
#include <sycl/queue.hpp>

namespace sycl {
inline namespace _V1 {

namespace detail {
class physical_mem_impl;
} // namespace detail

namespace ext::oneapi::experimental {

enum class address_access_mode : char { none = 0, read = 1, read_write = 2 };

class __SYCL_EXPORT physical_mem
    : public sycl::detail::OwnerLessBase<physical_mem> {
public:
  physical_mem(const device &SyclDevice, const context &SyclContext,
               size_t NumBytes);

  physical_mem(const queue &SyclQueue, size_t NumBytes)
      : physical_mem(SyclQueue.get_device(), SyclQueue.get_context(),
                     NumBytes) {}

  physical_mem(const physical_mem &rhs) = default;
  physical_mem(physical_mem &&rhs) = default;

  physical_mem &operator=(const physical_mem &rhs) = default;
  physical_mem &operator=(physical_mem &&rhs) = default;

  ~physical_mem() noexcept(false) {};

  bool operator==(const physical_mem &rhs) const { return impl == rhs.impl; }
  bool operator!=(const physical_mem &rhs) const { return !(*this == rhs); }

  void *map(uintptr_t Ptr, size_t NumBytes, address_access_mode Mode,
            size_t Offset = 0) const;

  context get_context() const;
  device get_device() const;

  size_t size() const noexcept;

private:
  std::shared_ptr<sycl::detail::physical_mem_impl> impl;

  template <class Obj>
  friend const decltype(Obj::impl) &
  sycl::detail::getSyclObjImpl(const Obj &SyclObject);

  template <class T>
  friend T sycl::detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);
};

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl

namespace std {
template <> struct hash<sycl::ext::oneapi::experimental::physical_mem> {
  size_t operator()(
      const sycl::ext::oneapi::experimental::physical_mem &PhysicalMem) const {
    return hash<std::shared_ptr<sycl::detail::physical_mem_impl>>()(
        sycl::detail::getSyclObjImpl(PhysicalMem));
  }
};
} // namespace std
