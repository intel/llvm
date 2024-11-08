//==------------- CompileTarget.cpp --- CompileTarget unit test ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define SYCL2020_DISABLE_DEPRECATION_WARNINGS

#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

using namespace sycl;

namespace sycl {
inline namespace _V1 {
namespace unittest {
static inline MockDeviceImage
generateImageWithCompileTarget(std::string KernelName,
                               std::string CompileTarget) {
  std::vector<char> Data(8 + CompileTarget.size());
  std::copy(CompileTarget.begin(), CompileTarget.end(), Data.data() + 8);
  MockProperty CompileTargetProperty("compile_target", Data,
                                     SYCL_PROPERTY_TYPE_BYTE_ARRAY);
  MockPropertySet PropSet;
  PropSet.insert(__SYCL_PROPERTY_SET_SYCL_DEVICE_REQUIREMENTS,
                 std::move(CompileTargetProperty));

  std::vector<unsigned char> Bin(CompileTarget.begin(), CompileTarget.end());
  // Null terminate the data so it can be interpreted as c string.
  Bin.push_back(0);

  std::vector<MockOffloadEntry> Entries = makeEmptyKernels({KernelName});

  auto DeviceTargetSpec = CompileTarget == "spir64_x86_64"
                              ? __SYCL_DEVICE_BINARY_TARGET_SPIRV64_X86_64
                              : __SYCL_DEVICE_BINARY_TARGET_SPIRV64_GEN;

  MockDeviceImage Img{SYCL_DEVICE_BINARY_TYPE_NATIVE, // Format
                      DeviceTargetSpec,               // DeviceTargetSpec
                      "",                             // Compile options
                      "",                             // Link options
                      std::move(Bin),
                      std::move(Entries),
                      std::move(PropSet)};

  return Img;
}
} // namespace unittest
} // namespace _V1
} // namespace sycl

class SingleTaskKernel;
class NDRangeKernel;
class RangeKernel;
class NoDeviceKernel;
class JITFallbackKernel;

MOCK_INTEGRATION_HEADER(SingleTaskKernel)
MOCK_INTEGRATION_HEADER(NDRangeKernel)
MOCK_INTEGRATION_HEADER(RangeKernel)
MOCK_INTEGRATION_HEADER(NoDeviceKernel)
MOCK_INTEGRATION_HEADER(JITFallbackKernel)

static sycl::unittest::MockDeviceImage Img[] = {
    sycl::unittest::generateDefaultImage({"SingleTaskKernel"}),
    sycl::unittest::generateImageWithCompileTarget("SingleTaskKernel",
                                                   "spir64_x86_64"),
    sycl::unittest::generateImageWithCompileTarget("SingleTaskKernel",
                                                   "intel_gpu_pvc"),
    sycl::unittest::generateImageWithCompileTarget("SingleTaskKernel",
                                                   "intel_gpu_skl"),
    sycl::unittest::generateDefaultImage({"NDRangeKernel"}),
    sycl::unittest::generateImageWithCompileTarget("NDRangeKernel",
                                                   "spir64_x86_64"),
    sycl::unittest::generateImageWithCompileTarget("NDRangeKernel",
                                                   "intel_gpu_pvc"),
    sycl::unittest::generateImageWithCompileTarget("NDRangeKernel",
                                                   "intel_gpu_skl"),
    sycl::unittest::generateDefaultImage({"RangeKernel"}),
    sycl::unittest::generateImageWithCompileTarget("RangeKernel",
                                                   "spir64_x86_64"),
    sycl::unittest::generateImageWithCompileTarget("RangeKernel",
                                                   "intel_gpu_pvc"),
    sycl::unittest::generateImageWithCompileTarget("RangeKernel",
                                                   "intel_gpu_skl"),
    sycl::unittest::generateImageWithCompileTarget("NoDeviceKernel",
                                                   "intel_gpu_bdw"),
    sycl::unittest::generateDefaultImage({"JITFallbackKernel"}),
    sycl::unittest::generateImageWithCompileTarget("JITFallbackKernel",
                                                   "intel_gpu_bdw"),
};

static sycl::unittest::MockDeviceImageArray<std::size(Img)> ImgArray{Img};

struct MockDeviceData {
  int Ip;
  ur_device_type_t DeviceType;
  ur_device_handle_t getHandle() {
    return reinterpret_cast<ur_device_handle_t>(this);
  }
  static MockDeviceData *fromHandle(ur_device_handle_t handle) {
    return reinterpret_cast<MockDeviceData *>(handle);
  }
};

// IP are from IntelGPUArchitectures/IntelCPUArchitectures in
// sycl/source/detail/device_info.hpp
MockDeviceData MockDevices[] = {
    {0x02400009, UR_DEVICE_TYPE_GPU}, // Skl
    {0x030f0000, UR_DEVICE_TYPE_GPU}, // Pvc
    {0, UR_DEVICE_TYPE_CPU},          // X86
    {8, UR_DEVICE_TYPE_CPU},          // Spr
    {9, UR_DEVICE_TYPE_CPU},          // Gnr
};

static ur_result_t redefinedDeviceGet(void *pParams) {
  auto params = *static_cast<ur_device_get_params_t *>(pParams);
  if (*params.ppNumDevices) {
    **params.ppNumDevices = static_cast<uint32_t>(std::size(MockDevices));
    return UR_RESULT_SUCCESS;
  }

  if (*params.pphDevices) {
    assert(*params.pNumEntries <= std::size(MockDevices));
    for (uint32_t i = 0; i < *params.pNumEntries; ++i) {
      (*params.pphDevices)[i] = MockDevices[i].getHandle();
    }
  }

  return UR_RESULT_SUCCESS;
}

std::vector<std::string> createWithBinaryLog;
static ur_result_t redefinedProgramCreateWithBinary(void *pParams) {
  auto params = *static_cast<ur_program_create_with_binary_params_t *>(pParams);
  for (uint32_t i = 0; i < *params.pnumDevices; ++i)
    createWithBinaryLog.push_back(
        reinterpret_cast<const char *>(*params.pppBinaries[i]));
  return UR_RESULT_SUCCESS;
}

std::vector<std::string> createWithILLog;
static ur_result_t redefinedProgramCreateWithIL(void *pParams) {
  auto params = *static_cast<ur_program_create_with_il_params_t *>(pParams);
  createWithILLog.push_back(reinterpret_cast<const char *>(*params.ppIL));
  return UR_RESULT_SUCCESS;
}

static ur_result_t redefinedDeviceGetInfo(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);
  if (*params.ppropName == UR_DEVICE_INFO_IP_VERSION && *params.ppPropValue) {
    int &ret = *static_cast<int *>(*params.ppPropValue);
    ret = MockDeviceData::fromHandle(*params.phDevice)->Ip;
  }
  if (*params.ppropName == UR_DEVICE_INFO_TYPE) {
    if (*params.ppPropValue)
      *static_cast<ur_device_type_t *>(*params.ppPropValue) =
          MockDeviceData::fromHandle(*params.phDevice)->DeviceType;
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(ur_device_type_t);
  }
  return UR_RESULT_SUCCESS;
}

static ur_result_t redefinedDeviceSelectBinary(void *pParams) {
  auto params = *static_cast<ur_device_select_binary_params_t *>(pParams);
  auto target = MockDeviceData::fromHandle(*params.phDevice)->DeviceType ==
                        UR_DEVICE_TYPE_CPU
                    ? UR_DEVICE_BINARY_TARGET_SPIRV64_X86_64
                    : UR_DEVICE_BINARY_TARGET_SPIRV64_GEN;
  uint32_t fallback = *params.pNumBinaries;
  for (uint32_t i = 0; i < *params.pNumBinaries; ++i) {
    if (strcmp((*params.ppBinaries)[i].pDeviceTargetSpec, target) == 0) {
      **params.ppSelectedBinary = i;
      return UR_RESULT_SUCCESS;
    }
    if (strcmp((*params.ppBinaries)[i].pDeviceTargetSpec,
               UR_DEVICE_BINARY_TARGET_SPIRV64) == 0) {
      fallback = i;
    }
  }
  if (fallback != *params.pNumBinaries) {
    **params.ppSelectedBinary = fallback;
    return UR_RESULT_SUCCESS;
  }
  return UR_RESULT_ERROR_INVALID_BINARY;
}

namespace syclex = sycl::ext::oneapi::experimental;
auto archSelector(syclex::architecture arch) {
  return [=](const device &dev) {
    if (dev.get_info<syclex::info::device::architecture>() == arch) {
      return 1;
    }
    return -1;
  };
}

class CompileTargetTest : public testing::Test {
protected:
  sycl::unittest::UrMock<> Mock;
  CompileTargetTest() {
    mock::getCallbacks().set_before_callback("urProgramCreateWithBinary",
                                             &redefinedProgramCreateWithBinary);
    mock::getCallbacks().set_before_callback("urProgramCreateWithIL",
                                             &redefinedProgramCreateWithIL);
    mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                            &redefinedDeviceGetInfo);
    mock::getCallbacks().set_after_callback("urDeviceGet", &redefinedDeviceGet);
    mock::getCallbacks().set_after_callback("urDeviceSelectBinary",
                                            &redefinedDeviceSelectBinary);
  }
};

template <typename F>
void checkUsedImageWithCompileTarget(const char *compile_target, F &&f) {
  createWithBinaryLog.clear();
  createWithILLog.clear();
  ASSERT_EQ(createWithBinaryLog.size(), 0U) << compile_target;
  ASSERT_EQ(createWithILLog.size(), 0U) << compile_target;
  f();
  EXPECT_EQ(createWithILLog.size(), 0U) << compile_target;
  ASSERT_EQ(createWithBinaryLog.size(), 1U) << compile_target;
  EXPECT_EQ(createWithBinaryLog.back(), compile_target) << compile_target;
}

void launchSingleTaskKernel(queue q) {
  q.single_task<SingleTaskKernel>([]() {});
}

TEST_F(CompileTargetTest, SingleTask) {
  checkUsedImageWithCompileTarget("intel_gpu_skl", [&]() {
    launchSingleTaskKernel(
        queue{archSelector(syclex::architecture::intel_gpu_skl)});
  });

  checkUsedImageWithCompileTarget("intel_gpu_pvc", [&]() {
    launchSingleTaskKernel(
        queue{archSelector(syclex::architecture::intel_gpu_pvc)});
  });

  checkUsedImageWithCompileTarget("spir64_x86_64", [&]() {
    launchSingleTaskKernel(queue{archSelector(syclex::architecture::x86_64)});
  });

  checkUsedImageWithCompileTarget("spir64_x86_64", [&]() {
    launchSingleTaskKernel(
        queue{archSelector(syclex::architecture::intel_cpu_spr)});
  });

  checkUsedImageWithCompileTarget("spir64_x86_64", [&]() {
    launchSingleTaskKernel(
        queue{archSelector(syclex::architecture::intel_cpu_gnr)});
  });
}

void launchNDRangeKernel(queue q) {
  q.submit([&](handler &cgh) {
    cgh.parallel_for<NDRangeKernel>(nd_range<1>(1, 1), [](auto) {});
  });
}

TEST_F(CompileTargetTest, NDRangeKernel) {
  checkUsedImageWithCompileTarget("intel_gpu_skl", [&]() {
    launchNDRangeKernel(
        queue{archSelector(syclex::architecture::intel_gpu_skl)});
  });

  checkUsedImageWithCompileTarget("intel_gpu_pvc", [&]() {
    launchNDRangeKernel(
        queue{archSelector(syclex::architecture::intel_gpu_pvc)});
  });

  checkUsedImageWithCompileTarget("spir64_x86_64", [&]() {
    launchNDRangeKernel(queue{archSelector(syclex::architecture::x86_64)});
  });

  checkUsedImageWithCompileTarget("spir64_x86_64", [&]() {
    launchNDRangeKernel(
        queue{archSelector(syclex::architecture::intel_cpu_spr)});
  });

  checkUsedImageWithCompileTarget("spir64_x86_64", [&]() {
    launchNDRangeKernel(
        queue{archSelector(syclex::architecture::intel_cpu_gnr)});
  });
}

void launchRangeKernel(queue q) {
  q.submit([&](handler &cgh) {
    cgh.parallel_for<NDRangeKernel>(range<1>(1), [](auto) {});
  });
}

TEST_F(CompileTargetTest, RangeKernel) {
  checkUsedImageWithCompileTarget("intel_gpu_skl", [&]() {
    launchRangeKernel(queue{archSelector(syclex::architecture::intel_gpu_skl)});
  });

  checkUsedImageWithCompileTarget("intel_gpu_pvc", [&]() {
    launchRangeKernel(queue{archSelector(syclex::architecture::intel_gpu_pvc)});
  });

  checkUsedImageWithCompileTarget("spir64_x86_64", [&]() {
    launchRangeKernel(queue{archSelector(syclex::architecture::x86_64)});
  });

  checkUsedImageWithCompileTarget("spir64_x86_64", [&]() {
    launchRangeKernel(queue{archSelector(syclex::architecture::intel_cpu_spr)});
  });

  checkUsedImageWithCompileTarget("spir64_x86_64", [&]() {
    launchRangeKernel(queue{archSelector(syclex::architecture::intel_cpu_gnr)});
  });
}

TEST_F(CompileTargetTest, NoDeviceKernel) {
  try {
    queue{}.single_task<NoDeviceKernel>([]() {});
  } catch (sycl::exception &e) {
    ASSERT_EQ(e.what(),
              std::string("No kernel named NoDeviceKernel was found"));
  }
}

TEST_F(CompileTargetTest, JITFallbackKernel) {
  createWithBinaryLog.clear();
  createWithILLog.clear();
  queue{}.single_task<JITFallbackKernel>([]() {});
  EXPECT_EQ(createWithBinaryLog.size(), 0U);
  ASSERT_EQ(createWithILLog.size(), 1U);
  EXPECT_EQ(createWithILLog.back(), "JITFallbackKernel");
}
