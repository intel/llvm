//==--- register_host_memory.hpp - SYCL host memory registration extension -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/export.hpp> // for __SYCL_EXPORT
#include <sycl/ext/oneapi/properties.hpp>

#include <cstddef> // for size_t
#include <cstdint> // for uint32_t
#include <type_traits>

namespace sycl {
inline namespace _V1 {
class context;

namespace ext::oneapi::experimental {

// Indicates that device code will only read from the registered range. Device
// writes to a range registered with this property are undefined behavior.
struct read_only_key : detail::compile_time_property_key<
                           detail::PropKind::RegisterHostMemoryReadOnly> {
  using value_t = property_value<read_only_key>;
};

inline constexpr read_only_key::value_t read_only;

namespace detail {
// Implementation-internal flags describing a host memory registration. They
// are translated to UR flags in the runtime library.
enum register_host_memory_flags : uint32_t {
  register_host_memory_flag_read_only = 1u << 0,
};

// Non-templated implementation entry points, defined in the SYCL runtime
// library. Flags is a bitwise OR of register_host_memory_flags values.
__SYCL_EXPORT void register_host_memory(void *Ptr, size_t NumBytes,
                                        const context &Ctxt, uint32_t Flags);
__SYCL_EXPORT void unregister_host_memory(void *Ptr, const context &Ctxt);

// Lowers a compile-time property list to the runtime flags word.
template <typename Properties> uint32_t getRegisterHostMemoryFlags() {
  uint32_t Flags = 0;
  if constexpr (std::decay_t<Properties>::template has_property<
                    read_only_key>())
    Flags |= register_host_memory_flag_read_only;
  return Flags;
}
} // namespace detail

/// Registers the existing host memory range \p ptr of \p numBytes bytes with
/// \p ctxt so that it behaves like a USM host allocation. See
/// sycl_ext_oneapi_register_host_memory for the full semantics.
///
/// \p ptr and \p numBytes must both be aligned to the host page size, \p ptr
/// must not be null, \p numBytes must not be zero, and every device in \p ctxt
/// must have aspect::ext_oneapi_register_host_memory.
template <typename Properties = empty_properties_t>
std::enable_if_t<is_property_list_v<std::decay_t<Properties>>>
register_host_memory(void *ptr, size_t numBytes, const context &ctxt,
                     Properties props = {}) {
  (void)props;
  detail::register_host_memory(
      ptr, numBytes, ctxt, detail::getRegisterHostMemoryFlags<Properties>());
}

/// Unregisters a host memory range previously registered with
/// register_host_memory. \p ptr must be the exact base pointer that was passed
/// to register_host_memory with the same \p ctxt, and the registration must
/// still be in effect. This does not free or unmap the underlying host memory.
inline void unregister_host_memory(void *ptr, const context &ctxt) {
  detail::unregister_host_memory(ptr, ctxt);
}

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
