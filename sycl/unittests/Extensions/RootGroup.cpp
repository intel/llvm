//==-------------- RootGroup.cpp - root group extension test ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>
#include <sycl/ext/oneapi/experimental/root_group.hpp>

// Include helpers for device image, kernel info, and Unified Runtime (UR) mocks
#include "helpers/MockDeviceImage.hpp"
#include "helpers/MockKernelInfo.hpp"
#include "helpers/UrMock.hpp"

// Define a mock kernel class with several operator() overloads for different
// SYCL item types
class QueryKernel {
public:
  void operator()() const {}
  void operator()(sycl::item<1>) const {}
  void operator()(sycl::nd_item<1> Item) const {}
};

// Specialize KernelInfo for QueryKernel to provide mock metadata for the kernel
namespace sycl {
inline namespace _V1 {
namespace detail {
template <>
struct KernelInfo<QueryKernel> : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() { return "QueryKernel"; }
  static constexpr int64_t getKernelSize() { return sizeof(QueryKernel); }
  static constexpr const char *getFileName() { return "QueryKernel.hpp"; }
  static constexpr const char *getFunctionName() {
    return "QueryKernelFunctionName";
  }
  static constexpr unsigned getLineNumber() { return 1; }
  static constexpr unsigned getColumnNumber() { return 1; }
};
} // namespace detail
} // namespace _V1
} // namespace sycl

// Test that querying max_num_work_groups with an invalid (zero) work-group size
// throws the correct exception
TEST(RootGroupTests, InvalidWorkGroupSize) {
  namespace syclex = sycl::ext::oneapi::experimental;

  // Create a mock device image containing the QueryKernel
  sycl::unittest::MockDeviceImage Img =
      sycl::unittest::generateDefaultImage({"QueryKernel"});
  const sycl::unittest::MockDeviceImageArray<1> ImgArray{&Img};
  const sycl::unittest::UrMock<> Mock;

  const sycl::queue q;
  // Get the kernel bundle and kernel object for QueryKernel
  const auto bundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context());
  const auto kernel = bundle.get_kernel<QueryKernel>();
  try {
    // Attempt to query max_num_work_groups with a zero work-group size
    kernel.ext_oneapi_get_info<
        syclex::info::kernel_queue_specific::max_num_work_groups>(q, {0}, 0);
    FAIL() << "The ext_oneapi_get_info query should have thrown.";
  } catch (const sycl::exception &e) {
    // Check that the correct error code and message are returned
    EXPECT_EQ(e.code(), sycl::make_error_code(sycl::errc::invalid));
    EXPECT_STREQ(e.what(), "The launch work-group size cannot be zero.");
  }
}

// Test that querying max_num_work_groups with a valid work-group size returns
// the expected value
TEST(RootGroupTests, ValidNumWorkGroupsQuery) {
  namespace syclex = sycl::ext::oneapi::experimental;

  // Create a mock device image containing the QueryKernel
  sycl::unittest::MockDeviceImage Img =
      sycl::unittest::generateDefaultImage({"QueryKernel"});
  const sycl::unittest::MockDeviceImageArray<1> ImgArray{&Img};
  const sycl::unittest::UrMock<> Mock;

  // Set up a mock callback to return a specific group count when queried
  constexpr std::size_t mock_group_count = 42;
  mock::getCallbacks().set_replace_callback(
      "urKernelSuggestMaxCooperativeGroupCountExp", [](void *pParams) {
        auto params = static_cast<
            ur_kernel_suggest_max_cooperative_group_count_params_t *>(pParams);
        **params->ppGroupCountRet = mock_group_count;
        return UR_RESULT_SUCCESS;
      });

  const sycl::queue q;
  // Get the kernel bundle and kernel object for QueryKernel
  const auto bundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context());
  const auto kernel = bundle.get_kernel<QueryKernel>();
  // Query max_num_work_groups with a valid work-group size (1)
  const auto maxWGs = kernel.ext_oneapi_get_info<
      syclex::info::kernel_queue_specific::max_num_work_groups>(q, {1}, 0);
  // Check that the returned value matches the mock group count
  EXPECT_EQ(maxWGs, mock_group_count);
}
