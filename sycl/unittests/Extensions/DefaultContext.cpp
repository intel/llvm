//==--------------------- DefaultContext.cpp -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <detail/config.hpp>
#include <helpers/PiMock.hpp>
#include <helpers/ScopedEnvVar.hpp>

#include <gtest/gtest.h>

// Same as defined in config.def
inline constexpr auto EnableDefaultContextsName =
    "SYCL_ENABLE_DEFAULT_CONTEXTS";

void test_default_context_enabled() {
  sycl::unittest::PiMock Mock1;
  sycl::platform Plt1 = Mock1.getPlatform();

  sycl::unittest::PiMock Mock2;
  sycl::platform Plt2 = Mock2.getPlatform();

  const sycl::device Dev1 = Plt1.get_devices()[0];
  const sycl::device Dev2 = Plt2.get_devices()[0];

  sycl::queue Queue1{Dev1};
  sycl::queue Queue2{Dev2};

  ASSERT_EQ(Queue1.get_context(), Queue2.get_context());

  ASSERT_EQ(Dev1.get_platform().ext_oneapi_get_default_context(),
            Dev2.get_platform().ext_oneapi_get_default_context());
}

void test_default_context_disabled() {
  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();

  bool catchException = false;
  try {
    (void)Plt.ext_oneapi_get_default_context();
  } catch (const std::exception &) {
    catchException = true;
  }

  ASSERT_TRUE(catchException)
      << "ext_oneapi_get_default_context did not throw an exception";
}

TEST(DefaultContextTest, DefaultContextTest) {
  using namespace sycl::detail;
  using namespace sycl::unittest;
  ScopedEnvVar var(EnableDefaultContextsName, "1",
                   SYCLConfig<SYCL_ENABLE_DEFAULT_CONTEXTS>::reset);

  test_default_context_enabled();
}

TEST(DefaultContextTest, DefaultContextCanBeDisabled) {
  using namespace sycl::detail;
  using namespace sycl::unittest;
  ScopedEnvVar var(EnableDefaultContextsName, "0",
                   SYCLConfig<SYCL_ENABLE_DEFAULT_CONTEXTS>::reset);

  test_default_context_disabled();
}

TEST(DefaultContextTest, DefaultContextCanBeDisabledEnabled) {
  sycl::detail::enable_ext_oneapi_default_context(false);
  test_default_context_disabled();

  sycl::detail::enable_ext_oneapi_default_context(true);
  test_default_context_enabled();
}

TEST(DefaultContextTest, DefaultContextValueChangedAfterQueueCreated) {
  sycl::detail::enable_ext_oneapi_default_context(false);

  sycl::unittest::PiMock Mock1;
  sycl::platform Plt = Mock1.getPlatform();

  const sycl::device Dev1 = Plt.get_devices()[0];
  const sycl::device Dev2 = Plt.get_devices()[0];
  const sycl::device Dev3 = Plt.get_devices()[0];

  sycl::queue Queue1{Dev1};

  sycl::detail::enable_ext_oneapi_default_context(true);

  sycl::queue Queue2{Dev2};

  ASSERT_NE(Queue1.get_context(), Queue2.get_context());

  sycl::queue Queue3{Dev3};

  ASSERT_EQ(Queue2.get_context(), Queue3.get_context());
}
