//==-------- ErrcUnbouncPlaceholder.cpp ------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#define SYCL2020_DISABLE_DEPRECATION_WARNINGS

#include <helpers/CommonRedefinitions.hpp>
#include <helpers/PiMock.hpp>
#include <helpers/TestKernel.hpp>

#include <CL/sycl.hpp>
#include <CL/sycl/accessor.hpp>

#include <gtest/gtest.h>

static pi_result redefinedDeviceGetInfo(pi_device device,
                                        pi_device_info param_name,
                                        size_t param_value_size,
                                        void *param_value,
                                        size_t *param_value_size_ret) {

  return PI_SUCCESS;
}

class SchedulerErrcTestClass : public ::testing::Test {
public:
  SchedulerErrcTestClass() : Plt{sycl::default_selector()} {}

protected:
  void SetUp() override {

    if (Plt.is_host()) {
      return;
    }

    Mock = std::make_unique<sycl::unittest::PiMock>(Plt);

    Mock->redefine<sycl::detail::PiApiKind::piDeviceGetInfo>(
        redefinedDeviceGetInfo); // reqd
  }

protected:
  std::unique_ptr<sycl::unittest::PiMock> Mock;
  sycl::platform Plt;
};

// placeholder accessor exception  // SYCL2020 4.7.6.9
TEST_F(SchedulerErrcTestClass,
       SchedulerErrcTest) { // mocking doesn't support host device
  if (Plt.is_host()) {
    GTEST_SKIP() << "unit tests with mocks not supported on host device\n";
    return;
  }
  sycl::queue q;
  // even if in the future mocks are supported on host,
  // host device executes kernels via a different method and there
  // is no good way to throw an exception at this time.
  if (!q.is_host()) {
    sycl::range<1> r(4);
    sycl::buffer<int, 1> b(r);
    try {
      sycl::accessor<int, 1, sycl::access::mode::read_write,
                     sycl::access::target::device,
                     sycl::access::placeholder::true_t>
          acc(b);

      q.submit([&](sycl::handler &cgh) {
        // we do NOT call .require(acc) without which we should throw a
        // synchronous exception with errc::kernel_argument
        cgh.parallel_for<TestKernel>(
            r, [=](sycl::id<1> index) { acc[index] = 0; });
      });
      q.wait_and_throw();
      FAIL() << "we should not be here, missing exception";
    } catch (sycl::exception &e) {
      EXPECT_EQ(e.code(), sycl::errc::kernel_argument)
          << "incorrect error code" << e.what();
      // "No kernel named  was found -46 (CL_INVALID_KERNEL_NAME)"
    } catch (...) {
      FAIL() << "some other exception encountered";
    }
  }
}
