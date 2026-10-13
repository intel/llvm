//==------------------------ RegisterHostMemory.cpp ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Test that sycl_ext_oneapi_register_host_memory validates its arguments,
// honors the device aspect, and calls the UR host memory registration APIs
// with the correct arguments.

#include <gtest/gtest.h>

#include <helpers/UrMock.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/detail/os_util.hpp>
#include <sycl/ext/oneapi/experimental/register_host_memory.hpp>
#include <sycl/usm.hpp>

#include <cstring>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

using namespace sycl;
namespace syclexp = sycl::ext::oneapi::experimental;

namespace {

static size_t getHostPageSize() {
#ifdef _WIN32
  SYSTEM_INFO Info;
  GetSystemInfo(&Info);
  return static_cast<size_t>(Info.dwPageSize);
#else
  return static_cast<size_t>(sysconf(_SC_PAGESIZE));
#endif
}

// Whether the mock device should advertise support for host memory
// registration via aspect::ext_oneapi_register_host_memory.
thread_local bool DeviceSupportsRegister = true;

// Captured arguments of the most recent UR register/unregister call.
thread_local void *LastRegisterPtr = nullptr;
thread_local size_t LastRegisterSize = 0;
thread_local void *LastUnregisterPtr = nullptr;
thread_local int RegisterCallCount = 0;
thread_local int UnregisterCallCount = 0;

// Registration flags captured from the most recent register call.
thread_local ur_exp_usm_host_alloc_register_flags_t LastRegisterFlags = 0;

// Result code the register/unregister mock should return, to exercise the
// UR-result-to-errc mapping in the runtime.
thread_local ur_result_t RegisterResult = UR_RESULT_SUCCESS;
thread_local ur_result_t UnregisterResult = UR_RESULT_SUCCESS;

ur_result_t redefinedDeviceGetInfo(void *pParams) {
  auto Params = *static_cast<ur_device_get_info_params_t *>(pParams);
  if (*Params.ppropName == UR_DEVICE_INFO_USM_HOST_ALLOC_REGISTER_SUPPORT_EXP) {
    if (*Params.ppPropValue)
      *static_cast<ur_bool_t *>(*Params.ppPropValue) = DeviceSupportsRegister;
    if (*Params.ppPropSizeRet)
      **Params.ppPropSizeRet = sizeof(ur_bool_t);
    return UR_RESULT_SUCCESS;
  }
  return sycl::unittest::MockAdapter::mock_urDeviceGetInfo(pParams);
}

ur_result_t redefinedHostAllocRegister(void *pParams) {
  auto Params =
      *static_cast<ur_usm_host_alloc_register_exp_params_t *>(pParams);
  LastRegisterPtr = *Params.ppHostMem;
  LastRegisterSize = *Params.psize;
  LastRegisterFlags = *Params.ppProperties ? (*Params.ppProperties)->flags : 0;
  ++RegisterCallCount;
  return RegisterResult;
}

ur_result_t redefinedHostAllocUnregister(void *pParams) {
  auto Params =
      *static_cast<ur_usm_host_alloc_unregister_exp_params_t *>(pParams);
  LastUnregisterPtr = *Params.ppHostMem;
  ++UnregisterCallCount;
  return UnregisterResult;
}

class RegisterHostMemoryTests : public ::testing::Test {
public:
  RegisterHostMemoryTests() : Mock{}, Ctxt{platform().get_devices()[0]} {}

protected:
  void SetUp() override {
    DeviceSupportsRegister = true;
    LastRegisterPtr = nullptr;
    LastRegisterSize = 0;
    LastUnregisterPtr = nullptr;
    RegisterCallCount = 0;
    UnregisterCallCount = 0;
    RegisterResult = UR_RESULT_SUCCESS;
    UnregisterResult = UR_RESULT_SUCCESS;
    LastRegisterFlags = 0;
    mock::getCallbacks().set_replace_callback("urDeviceGetInfo",
                                              &redefinedDeviceGetInfo);
    mock::getCallbacks().set_replace_callback("urUSMHostAllocRegisterExp",
                                              &redefinedHostAllocRegister);
    mock::getCallbacks().set_replace_callback("urUSMHostAllocUnregisterExp",
                                              &redefinedHostAllocUnregister);
  }

  unittest::UrMock<> Mock;
  context Ctxt;
};

// A successful registration forwards the exact pointer and size to UR and a
// matching unregistration forwards the same pointer.
TEST_F(RegisterHostMemoryTests, RegisterAndUnregisterForwardArguments) {
  const size_t PageSize = getHostPageSize();
  void *Ptr = detail::OSUtil::alignedAlloc(PageSize, PageSize);
  ASSERT_NE(Ptr, nullptr);

  syclexp::register_host_memory(Ptr, PageSize, Ctxt);
  EXPECT_EQ(RegisterCallCount, 1);
  EXPECT_EQ(LastRegisterPtr, Ptr);
  EXPECT_EQ(LastRegisterSize, PageSize);
  // No properties passed: no registration flags should be set.
  EXPECT_EQ(LastRegisterFlags, 0u);

  syclexp::unregister_host_memory(Ptr, Ctxt);
  EXPECT_EQ(UnregisterCallCount, 1);
  EXPECT_EQ(LastUnregisterPtr, Ptr);

  detail::OSUtil::alignedFree(Ptr);
}

// The read_only property is lowered to the UR read-only registration flag.
TEST_F(RegisterHostMemoryTests, ReadOnlyPropertyLowersToFlag) {
  const size_t PageSize = getHostPageSize();
  void *Ptr = detail::OSUtil::alignedAlloc(PageSize, PageSize);
  ASSERT_NE(Ptr, nullptr);

  syclexp::register_host_memory(Ptr, PageSize, Ctxt,
                                syclexp::properties{syclexp::read_only});
  EXPECT_EQ(RegisterCallCount, 1);
  EXPECT_TRUE(LastRegisterFlags &
              UR_EXP_USM_HOST_ALLOC_REGISTER_FLAG_READ_ONLY);

  syclexp::unregister_host_memory(Ptr, Ctxt);
  EXPECT_EQ(UnregisterCallCount, 1);

  detail::OSUtil::alignedFree(Ptr);
}

// A null pointer is rejected with errc::invalid before reaching UR.
TEST_F(RegisterHostMemoryTests, NullPointerThrowsInvalid) {
  try {
    syclexp::register_host_memory(nullptr, 4096, Ctxt);
    FAIL() << "Expected an exception.";
  } catch (const sycl::exception &E) {
    EXPECT_EQ(E.code(), make_error_code(errc::invalid));
  }
  EXPECT_EQ(RegisterCallCount, 0);
}

// A zero size is rejected with errc::invalid before reaching UR.
TEST_F(RegisterHostMemoryTests, ZeroSizeThrowsInvalid) {
  int Storage = 0;
  try {
    syclexp::register_host_memory(&Storage, 0, Ctxt);
    FAIL() << "Expected an exception.";
  } catch (const sycl::exception &E) {
    EXPECT_EQ(E.code(), make_error_code(errc::invalid));
  }
  EXPECT_EQ(RegisterCallCount, 0);
}

// Unregistering a null pointer is rejected with errc::invalid.
TEST_F(RegisterHostMemoryTests, UnregisterNullThrowsInvalid) {
  try {
    syclexp::unregister_host_memory(nullptr, Ctxt);
    FAIL() << "Expected an exception.";
  } catch (const sycl::exception &E) {
    EXPECT_EQ(E.code(), make_error_code(errc::invalid));
  }
  EXPECT_EQ(UnregisterCallCount, 0);
}

// When no device in the context supports the feature, registration throws
// errc::feature_not_supported and does not reach UR.
TEST_F(RegisterHostMemoryTests, UnsupportedDeviceThrowsFeatureNotSupported) {
  DeviceSupportsRegister = false;
  const size_t PageSize = getHostPageSize();
  void *Ptr = detail::OSUtil::alignedAlloc(PageSize, PageSize);
  ASSERT_NE(Ptr, nullptr);
  try {
    syclexp::register_host_memory(Ptr, PageSize, Ctxt);
    FAIL() << "Expected an exception.";
  } catch (const sycl::exception &E) {
    EXPECT_EQ(E.code(), make_error_code(errc::feature_not_supported));
  }
  EXPECT_EQ(RegisterCallCount, 0);
  detail::OSUtil::alignedFree(Ptr);
}

// The runtime maps a UR INVALID_VALUE result from either registration API to
// errc::invalid. The result is injected via the mock to test the mapping in
// isolation.
TEST_F(RegisterHostMemoryTests, BackendInvalidValueMapsToInvalid) {
  const size_t PageSize = getHostPageSize();
  void *Ptr = detail::OSUtil::alignedAlloc(PageSize, PageSize);
  ASSERT_NE(Ptr, nullptr);

  RegisterResult = UR_RESULT_ERROR_INVALID_VALUE;
  try {
    syclexp::register_host_memory(Ptr, PageSize, Ctxt);
    FAIL() << "Expected an exception.";
  } catch (const sycl::exception &E) {
    EXPECT_EQ(E.code(), make_error_code(errc::invalid));
  }

  UnregisterResult = UR_RESULT_ERROR_INVALID_VALUE;
  try {
    syclexp::unregister_host_memory(Ptr, Ctxt);
    FAIL() << "Expected an exception.";
  } catch (const sycl::exception &E) {
    EXPECT_EQ(E.code(), make_error_code(errc::invalid));
  }

  detail::OSUtil::alignedFree(Ptr);
}

} // namespace
