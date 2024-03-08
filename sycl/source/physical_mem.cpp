//==--- physical_mem.cpp - sycl_ext_oneapi_virtual_mem physical_mem class --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/physical_mem_impl.hpp>
#include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

physical_mem::physical_mem(const device &SyclDevice, const context &SyclContext,
                           size_t NumBytes) {
  if (!SyclDevice.has(aspect::ext_oneapi_virtual_mem))
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "Device does not support aspect::ext_oneapi_virtual_mem.");

  impl = std::make_shared<sycl::detail::physical_mem_impl>(
      SyclDevice, SyclContext, NumBytes);
}

void *physical_mem::map(uintptr_t Ptr, size_t NumBytes,
                        address_access_mode Mode, size_t Offset) const {
  return impl->map(Ptr, NumBytes, Mode, Offset);
}

context physical_mem::get_context() const { return impl->get_context(); }
device physical_mem::get_device() const { return impl->get_device(); }
size_t physical_mem::size() const noexcept { return impl->size(); }

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
