//==-- async_work_group_copy_ptr.hpp - OpenCL pointer conversion for AWG --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides convertToOpenCLGroupAsyncCopyPtr, which converts a sycl::multi_ptr
// to the decorated pointer type expected by __spirv_GroupAsyncCopy.
//
// Kept narrow so group.hpp and nd_item.hpp can use it without pulling in the
// full generic_type_traits.hpp (which transitively drags in aliases.hpp,
// bit_cast.hpp and the rest of the type-trait machinery).
//
// All dependencies (DecoratedType, access::address_space, multi_ptr) are
// already required by any header that does async_work_group_copy, so this
// header adds zero transitive cost.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/access/access.hpp>   // for DecoratedType, address_space
#include <sycl/detail/fwd/half.hpp> // for half_impl::BIsRepresentationT
#include <sycl/detail/fwd/multi_ptr.hpp>
#include <sycl/detail/type_traits/integer_traits.hpp> // for fixed_width_signed/unsigned

#include <cstddef>     // for std::byte
#include <stdint.h>    // for uint8_t, uint16_t
#include <type_traits> // for remove_const_t, is_const_v

namespace sycl {
inline namespace _V1 {
template <typename DataT, int NumElements> class __SYCL_EBO vec;
namespace ext::oneapi {
class bfloat16;
}

namespace detail {

// Maps a SYCL element type to the OpenCL scalar type expected by
// __spirv_GroupAsyncCopy.
template <typename T, typename = void> struct async_copy_elem_type {
  using type = T;
};

template <typename T>
struct async_copy_elem_type<T, std::enable_if_t<std::is_integral_v<T>>> {
  using type =
      std::conditional_t<std::is_signed_v<T>, fixed_width_signed<sizeof(T)>,
                         fixed_width_unsigned<sizeof(T)>>;
};

#if (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
template <> struct async_copy_elem_type<std::byte> {
  using type = uint8_t;
};
#endif

template <> struct async_copy_elem_type<half> {
  using type = half_impl::BIsRepresentationT;
};

template <> struct async_copy_elem_type<ext::oneapi::bfloat16> {
  // On host bfloat16 is left as-is; only rewrite to uint16_t on device,
  // mirroring the behaviour of convertToOpenCLType in generic_type_traits.hpp.
#ifdef __SYCL_DEVICE_ONLY__
  using type = uint16_t;
#else
  using type = ext::oneapi::bfloat16;
#endif
};

template <typename T, int N> struct async_copy_elem_type<vec<T, N>> {
  using elem = typename async_copy_elem_type<T>::type;
#ifdef __SYCL_DEVICE_ONLY__
  using type = std::conditional_t<N == 1, elem,
                                  elem __attribute__((ext_vector_type(N)))>;
#else
  using type = vec<elem, N>;
#endif
};

/// Convert a multi_ptr to the decorated raw pointer type expected by
/// __spirv_GroupAsyncCopy, rewriting the element type to its OpenCL equivalent.
template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress>
auto convertToOpenCLGroupAsyncCopyPtr(
    multi_ptr<ElementType, Space, DecorateAddress> Ptr) {
  using ElemNoCv = std::remove_const_t<ElementType>;
  using OpenCLElem = typename async_copy_elem_type<ElemNoCv>::type;
  using ConvertedElem = std::conditional_t<std::is_const_v<ElementType>,
                                           const OpenCLElem, OpenCLElem>;
  using ResultType = typename DecoratedType<ConvertedElem, Space>::type *;
  return reinterpret_cast<ResultType>(Ptr.get_decorated());
}

} // namespace detail
} // namespace _V1
} // namespace sycl
