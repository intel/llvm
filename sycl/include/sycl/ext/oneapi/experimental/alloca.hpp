//==--- alloca.hpp --- SYCL extension for private memory allocations--------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "sycl/exception.hpp"
#include "sycl/kernel_handler.hpp"
#include "sycl/multi_ptr.hpp"

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

#ifdef __SYCL_DEVICE_ONLY__

// On the device, this is an alias to __builtin_intel_sycl_alloca.

/// Function allocating and returning a pointer to an unitialized region of
/// memory capable of hosting `kh.get_specialization_constant<SizeSpecName>()`
/// elements of type \tp ElementType. The pointer will be a `sycl::private_ptr`
/// and will or will not be decorated depending on \tp DecorateAddres.
///
/// On the host, this function simply throws, as this is not supported there.
///
/// See sycl_ext_oneapi_private_alloca.
template <typename ElementType, auto &SizeSpecName,
          access::decorated DecorateAddress>
__SYCL_BUILTIN_ALIAS(__builtin_intel_sycl_alloca)
private_ptr<ElementType, DecorateAddress> private_alloca(kernel_handler &kh);

#else

// On the host, throw, this is not supported.
template <typename ElementType, auto &SizeSpecName,
          access::decorated DecorateAddress>
private_ptr<ElementType, DecorateAddress> private_alloca(kernel_handler &kh) {
  throw feature_not_supported("sycl::ext::oneapi::experimental::private_alloca "
                              "is not supported on host",
                              PI_ERROR_INVALID_OPERATION);
}

#endif // __SYCL_DEVICE_ONLY__

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
