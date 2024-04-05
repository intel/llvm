//==------------- dealloc.hpp - SYCL annotated usm deallocation ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/experimental/annotated_ptr/annotated_ptr.hpp>

namespace sycl {
inline namespace _V1 {
class queue;
namespace detail {
struct code_location;
}

void free(void *, const context &, const detail::code_location &);
void free(void *, const queue &, const detail::code_location &);

namespace ext {
namespace oneapi {
namespace experimental {

////
//  Deallocation
////
template <typename T, typename propList>
void free(annotated_ptr<T, propList> &ptr, const context &syclContext) {
  sycl::free(ptr.get(), syclContext);
}

template <typename T, typename propList>
void free(annotated_ptr<T, propList> &ptr, const queue &syclQueue) {
  sycl::free(ptr.get(), syclQueue);
}

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
