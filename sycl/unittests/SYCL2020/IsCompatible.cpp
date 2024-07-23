#include <sycl/sycl.hpp>

#include <helpers/MockKernelInfo.hpp>
#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>

#include <gtest/gtest.h>

class TestKernelCPU;
class TestKernelCPUInvalidReqdWGSize1D;
class TestKernelCPUInvalidReqdWGSize2D;
class TestKernelCPUInvalidReqdWGSize3D;
class TestKernelCPUValidReqdWGSize3D;
class TestKernelGPU;
class TestKernelACC;

MOCK_INTEGRATION_HEADER(TestKernelCPU)
MOCK_INTEGRATION_HEADER(TestKernelCPUInvalidReqdWGSize1D)
MOCK_INTEGRATION_HEADER(TestKernelCPUInvalidReqdWGSize2D)
MOCK_INTEGRATION_HEADER(TestKernelCPUInvalidReqdWGSize3D)
MOCK_INTEGRATION_HEADER(TestKernelCPUValidReqdWGSize3D)
MOCK_INTEGRATION_HEADER(TestKernelGPU)
MOCK_INTEGRATION_HEADER(TestKernelACC)

static sycl::unittest::PiImage
generateDefaultImage(std::initializer_list<std::string> KernelNames,
                     const std::vector<sycl::aspect> &Aspects, const std::vector<int> &ReqdWGSize = {}) {
  using namespace sycl::unittest;

  PiPropertySet PropSet;
  addDeviceRequirementsProps(PropSet, Aspects, ReqdWGSize);

  std::vector<unsigned char> Bin{0, 1, 2, 3, 4, 5}; // Random data

  PiArray<PiOffloadEntry> Entries = makeEmptyKernels(KernelNames);

  PiImage Img{PI_DEVICE_BINARY_TYPE_SPIRV,            // Format
              __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64, // DeviceTargetSpec
              "",                                     // Compile options
              "",                                     // Link options
              std::move(Bin),
              std::move(Entries),
              std::move(PropSet)};

  return Img;
}

static sycl::unittest::PiImage Imgs[7] = {
    // Images for validating checks based on max_work_group_size + aspects
    generateDefaultImage({"TestKernelCPU"}, {sycl::aspect::cpu},
                         {32}), // 32 <= 256 (OK)
    generateDefaultImage({"TestKernelCPUInvalidReqdWGSize1D"},
                         {sycl::aspect::cpu}, {257}), // 257 > 256 (FAIL)
    generateDefaultImage({"TestKernelCPUInvalidReqdWGSize2D"},
                         {sycl::aspect::cpu}, {32, 9}), // 32*9=288 > 256 (FAIL)
    // Images for validating checks based on max_work_item_sizes + aspects
    generateDefaultImage(
        {"TestKernelCPUInvalidReqdWGSize3D"}, {sycl::aspect::cpu},
        {4, 256, 6}), // 4 <= 254 (OK), 256 > 255 (FAIL), 6 <= 256 (OK)
    generateDefaultImage(
        {"TestKernelCPUValidReqdWGSize3D"}, {sycl::aspect::cpu},
        {2, 4, 5}), // 2 <= 254 (OK), 4 <= 255 (OK), 5 <= 256 (OK)
    // Images for validating checks for aspects
    generateDefaultImage({"TestKernelGPU"}, {sycl::aspect::gpu}),
    generateDefaultImage({"TestKernelACC"}, {sycl::aspect::accelerator})};

static sycl::unittest::PiImageArray<7> ImgArray{Imgs};

static pi_result redefinedDeviceGetInfoCPU(pi_device device,
                                           pi_device_info param_name,
                                           size_t param_value_size,
                                           void *param_value,
                                           size_t *param_value_size_ret) {
  if (param_name == PI_DEVICE_INFO_TYPE) {
    auto *Result = reinterpret_cast<_pi_device_type *>(param_value);
    *Result = PI_DEVICE_TYPE_CPU;
  }
  if (param_name == PI_DEVICE_INFO_MAX_WORK_GROUP_SIZE) {
    auto *Result = static_cast<size_t *>(param_value);
    *Result = 256;
  }
  if (param_name == PI_DEVICE_INFO_MAX_WORK_ITEM_SIZES) {
    auto *Result = static_cast<size_t *>(param_value);
    *Result = 256;
  }
  return PI_SUCCESS;
}

static pi_result redefinedDeviceGetInfoCPU3D(pi_device device,
                                           pi_device_info param_name,
                                           size_t param_value_size,
                                           void *param_value,
                                           size_t *param_value_size_ret) {
  if (param_name == PI_DEVICE_INFO_TYPE) {
    auto *Result = reinterpret_cast<_pi_device_type *>(param_value);
    *Result = PI_DEVICE_TYPE_CPU;
  }
  if (param_name == PI_DEVICE_INFO_MAX_WORK_GROUP_SIZE) {
    auto *Result = static_cast<size_t *>(param_value);
    *Result = 256;
  }
  if (param_name == PI_DEVICE_INFO_MAX_WORK_ITEM_SIZES) {
    auto *Result = static_cast<size_t *>(param_value);
    Result[0] = 256;
    Result[1] = 255;
    Result[2] = 254;
  }
  return PI_SUCCESS;
}

// Mock device is "GPU" by default, but we need to redefine it just in case
// if there are some changes in the future
static pi_result redefinedDeviceGetInfoGPU(pi_device device,
                                           pi_device_info param_name,
                                           size_t param_value_size,
                                           void *param_value,
                                           size_t *param_value_size_ret) {
  if (param_name == PI_DEVICE_INFO_TYPE) {
    auto *Result = reinterpret_cast<_pi_device_type *>(param_value);
    *Result = PI_DEVICE_TYPE_GPU;
  }
  return PI_SUCCESS;
}

static pi_result redefinedDeviceGetInfoACC(pi_device device,
                                           pi_device_info param_name,
                                           size_t param_value_size,
                                           void *param_value,
                                           size_t *param_value_size_ret) {
  if (param_name == PI_DEVICE_INFO_TYPE) {
    auto *Result = reinterpret_cast<_pi_device_type *>(param_value);
    *Result = PI_DEVICE_TYPE_ACC;
  }
  return PI_SUCCESS;
}

TEST(IsCompatible, CPU) {
  sycl::unittest::PiMock Mock;
  Mock.redefineAfter<sycl::detail::PiApiKind::piDeviceGetInfo>(
      redefinedDeviceGetInfoCPU);
  sycl::platform Plt = Mock.getPlatform();
  const sycl::device Dev = Plt.get_devices()[0];

  EXPECT_TRUE(Dev.is_cpu());
  EXPECT_TRUE(sycl::is_compatible<TestKernelCPU>(Dev));
  EXPECT_FALSE(sycl::is_compatible<TestKernelGPU>(Dev));
  EXPECT_FALSE(sycl::is_compatible<TestKernelACC>(Dev));
}

TEST(IsCompatible, CPUInvalidReqdWGSize1D) {
  sycl::unittest::PiMock Mock;
  Mock.redefineAfter<sycl::detail::PiApiKind::piDeviceGetInfo>(
      redefinedDeviceGetInfoCPU);
  sycl::platform Plt = Mock.getPlatform();
  const sycl::device Dev = Plt.get_devices()[0];

  EXPECT_FALSE(sycl::is_compatible<TestKernelCPUInvalidReqdWGSize1D>(Dev));
}

TEST(IsCompatible, CPUInvalidReqdWGSize2D) {
  sycl::unittest::PiMock Mock;
  Mock.redefineAfter<sycl::detail::PiApiKind::piDeviceGetInfo>(
      redefinedDeviceGetInfoCPU);
  sycl::platform Plt = Mock.getPlatform();
  const sycl::device Dev = Plt.get_devices()[0];

  EXPECT_FALSE(sycl::is_compatible<TestKernelCPUInvalidReqdWGSize2D>(Dev));
}

TEST(IsCompatible, CPUInvalidReqdWGSize3D) {
  sycl::unittest::PiMock Mock;
  Mock.redefineAfter<sycl::detail::PiApiKind::piDeviceGetInfo>(
      redefinedDeviceGetInfoCPU3D);
  sycl::platform Plt = Mock.getPlatform();
  const sycl::device Dev = Plt.get_devices()[0];

  EXPECT_FALSE(sycl::is_compatible<TestKernelCPUInvalidReqdWGSize3D>(Dev));
}

TEST(IsCompatible, CPUValidReqdWGSize3D) {
  sycl::unittest::PiMock Mock;
  Mock.redefineAfter<sycl::detail::PiApiKind::piDeviceGetInfo>(
      redefinedDeviceGetInfoCPU3D);
  sycl::platform Plt = Mock.getPlatform();
  const sycl::device Dev = Plt.get_devices()[0];

  EXPECT_TRUE(sycl::is_compatible<TestKernelCPUValidReqdWGSize3D>(Dev));
}

TEST(IsCompatible, GPU) {
  sycl::unittest::PiMock Mock;
  Mock.redefineAfter<sycl::detail::PiApiKind::piDeviceGetInfo>(
      redefinedDeviceGetInfoGPU);
  sycl::platform Plt = Mock.getPlatform();
  const sycl::device Dev = Plt.get_devices()[0];

  EXPECT_TRUE(Dev.is_gpu());
  EXPECT_FALSE(sycl::is_compatible<TestKernelCPU>(Dev));
  EXPECT_TRUE(sycl::is_compatible<TestKernelGPU>(Dev));
  EXPECT_FALSE(sycl::is_compatible<TestKernelACC>(Dev));
}

TEST(IsCompatible, ACC) {
  sycl::unittest::PiMock Mock;
  Mock.redefineAfter<sycl::detail::PiApiKind::piDeviceGetInfo>(
      redefinedDeviceGetInfoACC);
  sycl::platform Plt = Mock.getPlatform();
  const sycl::device Dev = Plt.get_devices()[0];

  EXPECT_TRUE(Dev.is_accelerator());
  EXPECT_FALSE(sycl::is_compatible<TestKernelCPU>(Dev));
  EXPECT_FALSE(sycl::is_compatible<TestKernelGPU>(Dev));
  EXPECT_TRUE(sycl::is_compatible<TestKernelACC>(Dev));
}
