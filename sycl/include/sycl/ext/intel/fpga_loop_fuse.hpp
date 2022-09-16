//==--------- fpga_loop_fuse.hpp --- SYCL FPGA Loop Fuse Extension ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

namespace ext::intel {

template <int N = 1, typename F>
void fpga_loop_fuse [[intel::loop_fuse(N)]] (F f) {
  f();
}

template <int N = 1, typename F>
void fpga_loop_fuse_independent [[intel::loop_fuse_independent(N)]] (F f) {
  f();
}

} // namespace ext::intel
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
