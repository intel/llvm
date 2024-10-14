//==--------------------- DefaultContext.cpp -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>
#include <helpers/TestKernel.hpp>
#include <helpers/UrMock.hpp>
#include <sycl/sycl.hpp>

// Check that the work group memory object is mapped to exactly one backend
// kernel argument.

namespace syclext = sycl::ext::oneapi::experimental;
using arg_type = syclext::work_group_memory<int, syclext::empty_properties_t>;

static int urKernelSetArgLocalCalls = 0;
inline ur_result_t redefined_urKernelSetArgLocal(void *) {
  ++urKernelSetArgLocalCalls;
  return UR_RESULT_SUCCESS;
}

TEST(URArgumentTest, URArgumentTest) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_replace_callback("urKernelSetArgLocal",
                                            &redefined_urKernelSetArgLocal);
  sycl::platform Platform = sycl::platform();
  const sycl::device dev = Platform.get_devices()[0];
  sycl::queue q{dev};
  syclext::submit(q, [&](sycl::handler &cgh) {
    arg_type data{cgh};
    const auto mykernel = [=](sycl::nd_item<1> it) { data = 42; };
    syclext::nd_launch<TestKernel<sizeof(mykernel)>>(
        cgh, sycl::nd_range<1>{1, 1}, mykernel);
  });
  q.wait();
  ASSERT_EQ(urKernelSetArgLocalCalls, 1);
}
