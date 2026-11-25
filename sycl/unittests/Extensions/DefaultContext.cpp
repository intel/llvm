//==--------------------- DefaultContext.cpp -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sycl/platform.hpp"
#include <sycl/sycl.hpp>

#include <detail/config.hpp>
#include <helpers/ScopedEnvVar.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

// Same as defined in config.def
inline constexpr auto EnableDefaultContextsName =
    "SYCL_ENABLE_DEFAULT_CONTEXTS";

void test_default_context_enabled() {
  sycl::unittest::UrMock<> Mock1;
  sycl::platform Plt1 = sycl::platform();

  sycl::unittest::UrMock<> Mock2;
  sycl::platform Plt2 = sycl::platform();

  const sycl::device Dev1 = Plt1.get_devices()[0];
  const sycl::device Dev2 = Plt2.get_devices()[0];

  sycl::queue Queue1{Dev1};
  sycl::queue Queue2{Dev2};

  ASSERT_EQ(Queue1.get_context(), Queue2.get_context());

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  ASSERT_EQ(Dev1.get_platform().ext_oneapi_get_default_context(),
            Dev2.get_platform().ext_oneapi_get_default_context());
#endif // __INTEL_PREVIEW_BREAKING_CHANGES

  ASSERT_EQ(Dev1.get_platform().khr_get_default_context(),
            Dev2.get_platform().khr_get_default_context());
}

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
void test_default_context_disabled() {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();

  bool catchException = false;
  try {
    (void)Plt.ext_oneapi_get_default_context();
  } catch (const std::exception &) {
    catchException = true;
  }

  ASSERT_TRUE(catchException)
      << "ext_oneapi_get_default_context did not throw an exception";
}
#endif // __INTEL_PREVIEW_BREAKING_CHANGES

TEST(DefaultContextTest, DefaultContextTest) {
  using namespace sycl::detail;
  using namespace sycl::unittest;
  ScopedEnvVar var(EnableDefaultContextsName, "1",
                   SYCLConfig<SYCL_ENABLE_DEFAULT_CONTEXTS>::reset);

  test_default_context_enabled();
}

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
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
#endif // __INTEL_PREVIEW_BREAKING_CHANGES

TEST(DefaultContextTest, DefaultContextValueChangedAfterQueueCreated) {
  sycl::detail::enable_ext_oneapi_default_context(false);

  sycl::unittest::UrMock<> Mock1;
  sycl::platform Plt = sycl::platform();

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

namespace {
const auto DEVICE_CPU = reinterpret_cast<ur_device_handle_t>(1u);
const auto DEVICE_GPU = reinterpret_cast<ur_device_handle_t>(2u);

ur_result_t redefine_urDeviceGet_CpuFirst(void *pParams) {
  auto params = *static_cast<ur_device_get_params_t *>(pParams);
  if (*params.ppNumDevices)
    **params.ppNumDevices = 2;
  if (*params.pphDevices) {
    if (*params.pNumEntries > 0)
      (*params.pphDevices)[0] = DEVICE_CPU;
    if (*params.pNumEntries > 1)
      (*params.pphDevices)[1] = DEVICE_GPU;
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t after_urDeviceGetInfo_CpuGpu(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);
  if (*params.ppropName != UR_DEVICE_INFO_TYPE)
    return UR_RESULT_SUCCESS;

  ur_device_type_t Type =
      *params.phDevice == DEVICE_CPU ? UR_DEVICE_TYPE_CPU : UR_DEVICE_TYPE_GPU;
  if (*params.ppPropValue)
    *static_cast<ur_device_type_t *>(*params.ppPropValue) = Type;
  if (*params.ppPropSizeRet)
    **params.ppPropSizeRet = sizeof(Type);
  return UR_RESULT_SUCCESS;
}

ur_device_handle_t FirstDeviceInUrContext = nullptr;

ur_result_t after_urContextCreate(void *pParams) {
  auto params = *static_cast<ur_context_create_params_t *>(pParams);
  if (*params.pDeviceCount > 0)
    FirstDeviceInUrContext = (*params.pphDevices)[0];
  return UR_RESULT_SUCCESS;
}
} // namespace

// The default context's first device must be the default queue's device, since
// backends treat the context's first device as the owner of e.g. images.
TEST(DefaultContextTest, FirstDeviceIsDefaultSelectedDevice) {
  using namespace sycl::detail;
  using namespace sycl::unittest;
  ScopedEnvVar var(EnableDefaultContextsName, "1",
                   SYCLConfig<SYCL_ENABLE_DEFAULT_CONTEXTS>::reset);

  UrMock<> Mock;
  mock::getCallbacks().set_replace_callback("urDeviceGet",
                                            &redefine_urDeviceGet_CpuFirst);
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &after_urDeviceGetInfo_CpuGpu);
  mock::getCallbacks().set_after_callback("urContextCreate",
                                          &after_urContextCreate);
  FirstDeviceInUrContext = nullptr;

  sycl::platform Plt = sycl::platform();
  // Precondition: the default-selected device is not first on the platform.
  ASSERT_EQ(Plt.get_devices().size(), 2u);
  ASSERT_TRUE(Plt.get_devices()[0].is_cpu());

  sycl::queue Q;
  ASSERT_TRUE(Q.get_device().is_gpu());

  ASSERT_EQ(Q.get_context(), Plt.khr_get_default_context());
  EXPECT_EQ(Q.get_context().get_devices()[0], Q.get_device());
  EXPECT_EQ(FirstDeviceInUrContext, DEVICE_GPU);
}
