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

class raw_kernel_arg {
public:
  raw_kernel_arg(const void *bytes, size_t count)
      : MArgData(bytes), MArgSize(count) {}

private:
  const void *MArgData;
  size_t MArgSize;

  friend class sycl::handler;
};

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
