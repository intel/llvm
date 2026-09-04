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

namespace detail {
class dynamic_parameter_impl;
struct RawKernelArgAccess;
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
  // For the enqueue paths that bind arguments without a handler
  friend struct detail::RawKernelArgAccess;
};

namespace detail {
// Helper for accessing the members of raw_kernel_arg.
struct RawKernelArgAccess {
  static const void *getData(const raw_kernel_arg &Arg) { return Arg.MArgData; }
  static size_t getSize(const raw_kernel_arg &Arg) { return Arg.MArgSize; }
};
} // namespace detail

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
