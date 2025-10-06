//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/access/access.hpp>
#include <sycl/detail/defines.hpp>

namespace sycl {
inline namespace _V1 {

template <typename DataT, int Dimensions, access::mode AccessMode,
          access::target AccessTarget, access::placeholder IsPlaceholder,
          typename PropertyListT>
class accessor;
template <typename DataT, int Dimensions, access::mode AccessMode>
class host_accessor;
template <typename DataT, int Dimensions>
class __SYCL_EBO
    __SYCL_SPECIAL_CLASS __SYCL_TYPE(local_accessor) local_accessor;

namespace detail {
template <typename DataT, int Dimensions, access::mode AccessMode,
          access::target AccessTarget, access::placeholder IsPlaceholder>
class image_accessor;
}

template <typename DataT, int Dimensions, access_mode AccessMode,
          image_target AccessTarget>
class unsampled_image_accessor;
template <typename DataT, int Dimensions, image_target AccessTarget>
class sampled_image_accessor;
template <typename DataT, int Dimensions, access_mode AccessMode>
class host_unsampled_image_accessor;
template <typename DataT, int Dimensions> class host_sampled_image_accessor;
} // namespace _V1
} // namespace sycl
