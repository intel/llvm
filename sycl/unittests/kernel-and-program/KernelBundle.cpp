//==--------- KernelBundle.cpp - Kernel bundle-related unit tests ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include <gtest/gtest.h>
#include <helpers/PiMock.hpp>
#include <sycl/sycl.hpp>

class KernelA;

TEST(KernelBundle, HasKernel) {
  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();
  
  const sycl::device Dev = Plt.get_devices()[0];

  sycl::queue Queue{Dev};

  const sycl::context Ctx = Queue.get_context();

  sycl::kernel_bundle KernelBundle =
    sycl::get_kernel_bundle<KernelA, sycl::bundle_state::executable>(Ctx);

  EXPECT_TRUE(KernelBundle.has_kernel<KernelA>());
}
