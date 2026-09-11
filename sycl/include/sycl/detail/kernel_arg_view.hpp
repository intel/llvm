//==---- kernel_arg_view.hpp --- SYCL kernel argument as bytes and kind ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/kernel_desc.hpp> // for kernel_param_kind_t

#include <stddef.h> // for size_t

namespace sycl {
inline namespace _V1 {
namespace detail {

inline namespace kernel_arg_view_v1 {

// A kernel argument reduced to what the runtime needs in order to bind it. Used
// by the enqueue functions that launch a `sycl::kernel` without building a
// command group, where the arguments are only known as bytes plus a kind. This
// is being passed across the ABI boundary, hence the versioned namespace.
struct KernelArgView {
  const void *MPtr;
  size_t MSize;
  kernel_param_kind_t MKind;
};

} // namespace kernel_arg_view_v1

} // namespace detail
} // namespace _V1
} // namespace sycl
