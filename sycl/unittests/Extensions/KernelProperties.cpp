//==----------------------- KernelProperties.cpp ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <helpers/PiMock.hpp>
#include <helpers/TestKernel.hpp>

#include <gtest/gtest.h>

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>

namespace oneapiext = sycl::ext::oneapi::experimental;

namespace {

inline pi_result after_piKernelGetInfo(pi_kernel kernel,
                                       pi_kernel_info param_name,
                                       size_t param_value_size,
                                       void *param_value,
                                       size_t *param_value_size_ret) {
  constexpr char MockKernel[] = "TestKernel";
  if (param_name == PI_KERNEL_INFO_FUNCTION_NAME) {
    if (param_value) {
      assert(param_value_size == sizeof(MockKernel));
      std::memcpy(param_value, MockKernel, sizeof(MockKernel));
    }
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(MockKernel);
  }
  return PI_SUCCESS;
}

class KernelPropertiesTests : public ::testing::Test {
public:
  KernelPropertiesTests()
      : Mock{}, Q{sycl::context(Mock.getPlatform()), sycl::default_selector_v} {
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
    Mock.redefineAfter<sycl::detail::PiApiKind::piKernelGetInfo>(
        after_piKernelGetInfo);
  }

  sycl::unittest::PiMock Mock;
  sycl::queue Q;
};

oneapiext::properties ForwardProgressProp{oneapiext::work_group_progress<
    oneapiext::forward_progress_guarantee::parallel,
    oneapiext::execution_scope::root_group>};

TEST_F(KernelPropertiesTests, SingleTaskKernelObjForwardProgress) {
  RunForwardProgressTest([&](sycl::kernel &K) {
    Q.submit(
        [&](sycl::handler &CGH) { CGH.single_task(ForwardProgressProp, K); });
  });
}

TEST_F(KernelPropertiesTests, ParallelForRange1DKernelObjForwardProgress) {
  RunForwardProgressTest([&](sycl::kernel &K) {
    Q.submit([&](sycl::handler &CGH) {
      CGH.parallel_for(sycl::range<1>{1}, ForwardProgressProp, K);
    });
  });
}

TEST_F(KernelPropertiesTests, ParallelForRange2DKernelObjForwardProgress) {
  RunForwardProgressTest([&](sycl::kernel &K) {
    Q.submit([&](sycl::handler &CGH) {
      CGH.parallel_for(sycl::range<2>{1, 1}, ForwardProgressProp, K);
    });
  });
}

TEST_F(KernelPropertiesTests, ParallelForRange3DKernelObjForwardProgress) {
  RunForwardProgressTest([&](sycl::kernel &K) {
    Q.submit([&](sycl::handler &CGH) {
      CGH.parallel_for(sycl::range<3>{1, 1, 1}, ForwardProgressProp, K);
    });
  });
}

TEST_F(KernelPropertiesTests, ParallelForNDRange1DKernelObjForwardProgress) {
  RunForwardProgressTest([&](sycl::kernel &K) {
    Q.submit([&](sycl::handler &CGH) {
      CGH.parallel_for(sycl::nd_range<1>{sycl::range<1>{1}, sycl::range<1>{1}},
                       ForwardProgressProp, K);
    });
  });
}

TEST_F(KernelPropertiesTests, ParallelForNDRange2DKernelObjForwardProgress) {
  RunForwardProgressTest([&](sycl::kernel &K) {
    Q.submit([&](sycl::handler &CGH) {
      CGH.parallel_for(
          sycl::nd_range<2>{sycl::range<2>{1, 1}, sycl::range<2>{1, 1}},
          ForwardProgressProp, K);
    });
  });
}

TEST_F(KernelPropertiesTests, ParallelForNDRange3DKernelObjForwardProgress) {
  RunForwardProgressTest([&](sycl::kernel &K) {
    Q.submit([&](sycl::handler &CGH) {
      CGH.parallel_for(
          sycl::nd_range<3>{sycl::range<3>{1, 1, 1}, sycl::range<3>{1, 1, 1}},
          ForwardProgressProp, K);
    });
  });
}

} // namespace
