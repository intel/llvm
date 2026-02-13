// RUN: %clangxx -fsycl -fsyntax-only %s
//=---queue_offset_shortcut_initlist.cpp - SYCL queue offset shortcuts test--=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

class KernelNameA;
class KernelNameB;
class KernelNameC;

int main() {
  sycl::queue q;
  sycl::event e;
  // Check that init-list works here.
  q.parallel_for<KernelNameA>(sycl::range<1>{1}, sycl::id<1>{0}, {e},
                              [=](sycl::item<1> i) {});
  q.parallel_for<KernelNameB>(sycl::range<2>{1, 1}, sycl::id<2>{0, 0}, {e},
                              [=](sycl::item<2> i) {});
  q.parallel_for<KernelNameC>(sycl::range<3>{1, 1, 1}, sycl::id<3>{0, 0, 0},
                              {e}, [=](sycl::item<3> i) {});
}
