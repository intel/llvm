//==--- raw_kernel_arg.hpp --- SYCL extension for raw kernel args ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <stddef.h>

namespace sycl {
inline namespace _V1 {

class handler;

namespace ext::oneapi::experimental {
class raw_kernel_arg;
} // namespace ext::oneapi::experimental

namespace detail {
class FreeFunctionArgCollector;
// Forward declaration of the free function argument setter overload for
// raw_kernel_arg so that raw_kernel_arg can befriend it below.
void setFreeFunctionArg(FreeFunctionArgCollector &, int,
                        ext::oneapi::experimental::raw_kernel_arg &&);
} // namespace detail

namespace ext::oneapi::experimental {

namespace detail {
class dynamic_parameter_impl;
} // namespace detail

class raw_kernel_arg {
public:
  raw_kernel_arg(const void *bytes, size_t count)
      : MArgData(bytes), MArgSize(count) {}

private:
  const void *MArgData;
  size_t MArgSize;

  friend class sycl::handler;
  // For sycl_ext_oneapi_graph integration
  friend class detail::dynamic_parameter_impl;
  // For free function kernel direct submission.
  friend void sycl::_V1::detail::setFreeFunctionArg(
      sycl::_V1::detail::FreeFunctionArgCollector &, int, raw_kernel_arg &&);
};

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
