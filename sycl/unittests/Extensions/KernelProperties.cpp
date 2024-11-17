//==----------------------- KernelProperties.cpp ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sycl/platform.hpp"
#include "ur_mock_helpers.hpp"
#include <helpers/UrMock.hpp>
#include <helpers/TestKernel.hpp>

#include <gtest/gtest.h>

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/kernel_bundle.hpp>

namespace oneapiext = sycl::ext::oneapi::experimental;

namespace {

inline ur_result_t after_urKernelGetInfo(void* pParams) {
  auto params = *static_cast<ur_kernel_get_info_params_t*>(pParams);
  constexpr char MockKernel[] = "TestKernel";
  if (*params.ppropName == UR_KERNEL_INFO_FUNCTION_NAME) {
    if (*params.ppPropValue) {
      assert(*params.ppropSize == sizeof(MockKernel));
      std::memcpy(*params.ppPropValue, MockKernel, sizeof(MockKernel));
    }
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(MockKernel);
  }
  return UR_RESULT_SUCCESS;
}

class KernelPropertiesTests : public ::testing::Test {
public:
  KernelPropertiesTests()
      : Mock{}, Q{sycl::context(sycl::platform()), sycl::default_selector_v} {
  }

  inline sycl::kernel GetTestKernel() {
    auto KB = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
        Q.get_context());
    return KB.get_kernel<TestKernel<>>();
  }

  template <typename FuncT> void RunForwardProgressTest(const FuncT &F) {
    sycl::kernel K = GetTestKernel();

    try {
      F(K);
      FAIL() << "Exception was expected.";
    } catch (sycl::exception &E) {
      // Intention of this test is to ensure that properties without kernel
      // effects are applied correctly through the interface. For forward
      // progress we can do this by expecting that the kernel reports that it is
      // using an unsupported forward progress.
      ASSERT_EQ(E.code(),
                sycl::make_error_code(sycl::errc::feature_not_supported));
    }
  }

protected:
  void SetUp() override {
    mock::getCallbacks().set_after_callback("urKernelGetInfo", after_urKernelGetInfo);
  }

  sycl::unittest::UrMock<> Mock;
  sycl::queue Q;
};

oneapiext::properties ForwardProgressProp{oneapiext::work_group_progress<
    oneapiext::forward_progress_guarantee::parallel,
    oneapiext::execution_scope::root_group>};

TEST_F(KernelPropertiesTests, ParallelFor1DKernelObjForwardProgress) {
  RunForwardProgressTest([&](sycl::kernel &K) {
    oneapiext::launch_config<sycl::range<1>, decltype(ForwardProgressProp)> LC{
        {1}, ForwardProgressProp};
    oneapiext::parallel_for(Q, LC, K);
  });
}

TEST_F(KernelPropertiesTests, ParallelFor2DKernelObjForwardProgress) {
  RunForwardProgressTest([&](sycl::kernel &K) {
    oneapiext::launch_config<sycl::range<2>, decltype(ForwardProgressProp)> LC{
        {1, 1}, ForwardProgressProp};
    oneapiext::parallel_for(Q, LC, K);
  });
}

TEST_F(KernelPropertiesTests, ParallelFor3DKernelObjForwardProgress) {
  RunForwardProgressTest([&](sycl::kernel &K) {
    oneapiext::launch_config<sycl::range<3>, decltype(ForwardProgressProp)> LC{
        {1, 1, 1}, ForwardProgressProp};
    oneapiext::parallel_for(Q, LC, K);
  });
}

TEST_F(KernelPropertiesTests, NDLaunch1DKernelObjForwardProgress) {
  RunForwardProgressTest([&](sycl::kernel &K) {
    oneapiext::launch_config<sycl::nd_range<1>, decltype(ForwardProgressProp)>
        LC{{{1}, {1}}, ForwardProgressProp};
    oneapiext::nd_launch(Q, LC, K);
  });
}

TEST_F(KernelPropertiesTests, NDLaunch2DKernelObjForwardProgress) {
  RunForwardProgressTest([&](sycl::kernel &K) {
    oneapiext::launch_config<sycl::nd_range<2>, decltype(ForwardProgressProp)>
        LC{{{1, 1}, {1, 1}}, ForwardProgressProp};
    oneapiext::nd_launch(Q, LC, K);
  });
}

TEST_F(KernelPropertiesTests, NDLaunch3DKernelObjForwardProgress) {
  RunForwardProgressTest([&](sycl::kernel &K) {
    oneapiext::launch_config<sycl::nd_range<3>, decltype(ForwardProgressProp)>
        LC{{{1, 1, 1}, {1, 1, 1}}, ForwardProgressProp};
    oneapiext::nd_launch(Q, LC, K);
  });
}

} // namespace
