//==--------- fpga_loop_fuse.hpp --- SYCL FPGA Loop Fuse Extension ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace intel {

template <int _N = 1, typename _F>
void fpga_loop_fuse [[intel::loop_fuse(_N)]] (_F f) {
  f();
}

template <int _N = 1, typename _F>
void fpga_loop_fuse_independent [[intel::loop_fuse_independent(_N)]] (_F f) {
  f();
}

} // namespace intel
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
