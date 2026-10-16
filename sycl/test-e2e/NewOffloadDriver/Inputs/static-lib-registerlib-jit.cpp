//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

#include "static-lib-registerlib.hpp"

void run_jit(std::size_t *out, std::size_t N) {
  sycl::queue q;
  std::size_t *p = sycl::malloc_shared<std::size_t>(N, q);
  q.parallel_for<class KernelJIT>(sycl::range<1>{N}, [=](sycl::id<1> idx) {
     p[idx] = idx[0] + JITOffset;
   }).wait();
  for (std::size_t i = 0; i < N; ++i)
    out[i] = p[i];
  sycl::free(p, q);
}
