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
#include <sycl/ext/oneapi/properties/properties.hpp>
#include <sycl/queue.hpp>

namespace sycl {
inline namespace _V1 {

namespace detail {
class physical_mem_impl;
} // namespace detail

namespace ext::oneapi::experimental {

enum class address_access_mode : char { none = 0, read = 1, read_write = 2 };

// Property controlling whether a physical_mem object can be shared across
// processes via IPC.
struct enable_ipc
    : detail::run_time_property_key<enable_ipc, detail::PropKind::EnableIpc> {
  enable_ipc(bool enable = true) : value(enable) {}
  bool value;
};

using enable_ipc_key = enable_ipc;

inline bool operator==(const enable_ipc &lhs, const enable_ipc &rhs) {
  return lhs.value == rhs.value;
}
inline bool operator!=(const enable_ipc &lhs, const enable_ipc &rhs) {
  return !(lhs == rhs);
}

class __SYCL_EXPORT physical_mem
    : public sycl::detail::OwnerLessBase<physical_mem> {
  friend sycl::detail::ImplUtils;

public:
  template <typename PropertyListT = empty_properties_t>
  physical_mem(const device &SyclDevice, const context &SyclContext,
               size_t NumBytes, PropertyListT Props = {})
      : physical_mem(SyclDevice, SyclContext, NumBytes, [&Props]() -> bool {
          if constexpr (PropertyListT::template has_property<enable_ipc_key>())
            return Props.template get_property<enable_ipc_key>().value;
          return false;
        }()) {}

  template <typename PropertyListT = empty_properties_t>
  physical_mem(const queue &SyclQueue, size_t NumBytes,
               PropertyListT Props = {})
      : physical_mem(SyclQueue.get_device(), SyclQueue.get_context(), NumBytes,
                     Props) {}

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

  bool ext_oneapi_ipc_enabled() const;
  bool ipc_enabled() const { return ext_oneapi_ipc_enabled(); }

  // Original public constructors preserved for ABI and source compatibility.
  physical_mem(const device &SyclDevice, const context &SyclContext,
               size_t NumBytes);
  physical_mem(const queue &SyclQueue, size_t NumBytes);

private:
  // Internal constructor used when IPC is requested via properties.
  physical_mem(const device &SyclDevice, const context &SyclContext,
               size_t NumBytes, bool EnableIpc);

  // Internal constructor for creating a physical_mem from an existing impl
  // (used by createSyclObjFromImpl, e.g. when opening from an IPC handle).
  explicit physical_mem(std::shared_ptr<sycl::detail::physical_mem_impl> Impl)
      : impl(std::move(Impl)) {}

  std::shared_ptr<sycl::detail::physical_mem_impl> impl;
};

template <>
struct is_property_key_of<enable_ipc_key, physical_mem> : std::true_type {};

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl

template <>
struct std::hash<sycl::ext::oneapi::experimental::physical_mem>
    : public sycl::detail::sycl_obj_hash<
          sycl::ext::oneapi::experimental::physical_mem> {};
