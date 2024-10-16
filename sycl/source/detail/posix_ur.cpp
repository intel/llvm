//==---------------- posix_ur.cpp ------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/defines_elementary.hpp>
#include <sycl/detail/iostream_proxy.hpp>
#include <sycl/detail/ur.hpp>

#include <dlfcn.h>
#include <string>

namespace sycl {
inline namespace _V1 {
namespace detail::ur {

void *getURLoaderLibrary() { return nullptr; }

} // namespace detail::ur
} // namespace _V1
} // namespace sycl
