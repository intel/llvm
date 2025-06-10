//==----------------------- SpillMemorySize.cpp ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>
#include <helpers/TestKernel.hpp>
#include <helpers/UrMock.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/sycl.hpp>
#include <vector>

using namespace sycl;

namespace {
ur_result_t redefinedKernelGetInfo(void *pParams) {
  constexpr size_t DeviceNum = 1;
  auto params = *static_cast<ur_kernel_get_info_params_t *>(pParams);

  if (*params.ppropName == UR_KERNEL_INFO_SPILL_MEM_SIZE) {
    if (*params.ppPropValue == nullptr) {
      size_t *ResultValuesNumber =
          reinterpret_cast<size_t *>(*params.ppPropSizeRet);
      *ResultValuesNumber = DeviceNum * sizeof(uint32_t);
    } else {
      constexpr uint32_t Device2SpillMap[] = {42};
      assert(*params.ppropSize == sizeof(Device2SpillMap));

      std::memcpy(*params.ppPropValue, Device2SpillMap,
                  sizeof(Device2SpillMap));
    }
  }
  return UR_RESULT_SUCCESS;
}
} // namespace

class KernelQueriesTests : public ::testing::Test {
public:
  KernelQueriesTests()
      : Mock{},
        Queue{sycl::context(sycl::platform()), sycl::default_selector_v} {}

  inline sycl::kernel GetTestKernel() {
    auto KB = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
        Queue.get_context());
    return KB.get_kernel<TestKernel<>>();
  }

protected:
  void SetUp() override {
    mock::getCallbacks().set_after_callback("urKernelGetInfo",
                                            &redefinedKernelGetInfo);
  }

  sycl::unittest::UrMock<backend::ext_oneapi_level_zero> Mock;
  sycl::queue Queue;
};

TEST(KernelQueriesBasicTests, NoAspect) {
  sycl::unittest::UrMock<> Mock;
  sycl::queue q{sycl::context(sycl::platform()), sycl::default_selector_v};
  auto KB =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context());
  auto kernel = KB.get_kernel<TestKernel<>>();
  const auto dev = q.get_device();
  try {
    kernel.template get_info<
        sycl::ext::intel::info::kernel_device_specific::spill_memory_size>(dev);
    FAIL() << "Exception was expected.";
  } catch (const sycl::exception &E) {
    // Intention of this test is to ensure that whenever a device does not
    // expose an aspect required by the query an exception is thrown.
    ASSERT_EQ(E.code(),
              sycl::make_error_code(sycl::errc::feature_not_supported));
  }
}

TEST_F(KernelQueriesTests, SpillMemorySize) {
  sycl::kernel kernel = GetTestKernel();
  const auto dev = Queue.get_device();
  const auto spillMemSz = kernel.get_info<
      sycl::ext::intel::info::kernel_device_specific::spill_memory_size>(dev);

  static_assert(std::is_same_v<std::remove_cv_t<decltype(spillMemSz)>, size_t>,
                "spill_memory_size query must return size_t");

  EXPECT_EQ(spillMemSz, size_t{42});
}
