#include <sycl/sycl.hpp>

#include <helpers/MockKernelInfo.hpp>
#include <helpers/PiImage.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

class TestKernelCPU;
class TestKernelCPUInvalidReqdWGSize1D;
class TestKernelCPUInvalidReqdWGSize2D;
class TestKernelCPUInvalidReqdWGSize3D;
class TestKernelCPUValidReqdWGSize3D;
class TestKernelGPU;
class TestKernelACC;

namespace sycl {
inline namespace _V1 {
namespace detail {
template <>
struct KernelInfo<TestKernelCPU> : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() { return "TestKernelCPU"; }
};

template <>
struct KernelInfo<TestKernelCPUInvalidReqdWGSize1D>
    : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() {
    return "TestKernelCPUInvalidReqdWGSize1D";
  }
};

template <>
struct KernelInfo<TestKernelCPUInvalidReqdWGSize2D>
    : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() {
    return "TestKernelCPUInvalidReqdWGSize2D";
  }
};

template <>
struct KernelInfo<TestKernelCPUInvalidReqdWGSize3D>
    : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() {
    return "TestKernelCPUInvalidReqdWGSize3D";
  }
};

template <>
struct KernelInfo<TestKernelCPUValidReqdWGSize3D>
    : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() {
    return "TestKernelCPUValidReqdWGSize3D";
  }
};

template <>
struct KernelInfo<TestKernelGPU> : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() { return "TestKernelGPU"; }
};

template <>
struct KernelInfo<TestKernelACC> : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() { return "TestKernelACC"; }
};

} // namespace detail
} // namespace _V1
} // namespace sycl

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

static ur_result_t redefinedDeviceGetInfoCPU(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);
  if (*params.ppropName == UR_DEVICE_INFO_TYPE) {
    auto *Result = reinterpret_cast<ur_device_type_t *>(*params.ppPropValue);
    *Result = UR_DEVICE_TYPE_CPU;
  }
  if (*params.ppropName == UR_DEVICE_INFO_MAX_WORK_GROUP_SIZE) {
    auto *Result = static_cast<size_t *>(*params.ppPropValue);
    *Result = 256;
  }
  if (*params.ppropName == UR_DEVICE_INFO_MAX_WORK_ITEM_SIZES) {
    auto *Result = static_cast<size_t *>(*params.ppPropValue);
    *Result = 256;
  }
  return UR_RESULT_SUCCESS;
}

static ur_result_t redefinedDeviceGetInfoCPU3D(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);
  if (*params.ppropName == UR_DEVICE_INFO_TYPE) {
    auto *Result = reinterpret_cast<ur_device_type_t *>(*params.ppPropValue);
    *Result = UR_DEVICE_TYPE_CPU;
  }
  if (*params.ppropName == UR_DEVICE_INFO_MAX_WORK_GROUP_SIZE) {
    auto *Result = static_cast<size_t *>(*params.ppPropValue);
    *Result = 256;
  }
  if (*params.ppropName == UR_DEVICE_INFO_MAX_WORK_ITEM_SIZES) {
    auto *Result = static_cast<size_t *>(*params.ppPropValue);
    Result[0] = 256;
    Result[1] = 255;
    Result[2] = 254;
  }
  return UR_RESULT_SUCCESS;
}

// Mock device is "GPU" by default, but we need to redefine it just in case
// if there are some changes in the future
static ur_result_t redefinedDeviceGetInfoGPU(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);
  if (*params.ppropName == UR_DEVICE_INFO_TYPE) {
    auto *Result = reinterpret_cast<ur_device_type_t *>(*params.ppPropValue);
    *Result = UR_DEVICE_TYPE_GPU;
  }
  return UR_RESULT_SUCCESS;
}

static ur_result_t redefinedDeviceGetInfoACC(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);
  if (*params.ppropName == UR_DEVICE_INFO_TYPE) {
    auto *Result = reinterpret_cast<ur_device_type_t *>(*params.ppPropValue);
    *Result = UR_DEVICE_TYPE_FPGA;
  }
  return UR_RESULT_SUCCESS;
}

TEST(IsCompatible, CPU) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &redefinedDeviceGetInfoCPU);
  sycl::platform Plt = sycl::platform();
  const sycl::device Dev = Plt.get_devices()[0];

  EXPECT_TRUE(Dev.is_cpu());
  EXPECT_TRUE(sycl::is_compatible<TestKernelCPU>(Dev));
  EXPECT_FALSE(sycl::is_compatible<TestKernelGPU>(Dev));
  EXPECT_FALSE(sycl::is_compatible<TestKernelACC>(Dev));
}

TEST(IsCompatible, CPUInvalidReqdWGSize1D) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &redefinedDeviceGetInfoCPU);
  sycl::platform Plt = sycl::platform();
  const sycl::device Dev = Plt.get_devices()[0];

  EXPECT_FALSE(sycl::is_compatible<TestKernelCPUInvalidReqdWGSize1D>(Dev));
}

TEST(IsCompatible, CPUInvalidReqdWGSize2D) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &redefinedDeviceGetInfoCPU);
  sycl::platform Plt = sycl::platform();
  const sycl::device Dev = Plt.get_devices()[0];

  EXPECT_FALSE(sycl::is_compatible<TestKernelCPUInvalidReqdWGSize2D>(Dev));
}

TEST(IsCompatible, CPUInvalidReqdWGSize3D) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &redefinedDeviceGetInfoCPU3D);
  sycl::platform Plt = sycl::platform();
  const sycl::device Dev = Plt.get_devices()[0];

  EXPECT_FALSE(sycl::is_compatible<TestKernelCPUInvalidReqdWGSize3D>(Dev));
}

TEST(IsCompatible, CPUValidReqdWGSize3D) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &redefinedDeviceGetInfoCPU3D);
  sycl::platform Plt = sycl::platform();
  const sycl::device Dev = Plt.get_devices()[0];

  EXPECT_TRUE(sycl::is_compatible<TestKernelCPUValidReqdWGSize3D>(Dev));
}

TEST(IsCompatible, GPU) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &redefinedDeviceGetInfoGPU);
  sycl::platform Plt = sycl::platform();
  const sycl::device Dev = Plt.get_devices()[0];

  EXPECT_TRUE(Dev.is_gpu());
  EXPECT_FALSE(sycl::is_compatible<TestKernelCPU>(Dev));
  EXPECT_TRUE(sycl::is_compatible<TestKernelGPU>(Dev));
  EXPECT_FALSE(sycl::is_compatible<TestKernelACC>(Dev));
}

TEST(IsCompatible, ACC) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &redefinedDeviceGetInfoACC);
  sycl::platform Plt = sycl::platform();
  const sycl::device Dev = Plt.get_devices()[0];

  EXPECT_TRUE(Dev.is_accelerator());
  EXPECT_FALSE(sycl::is_compatible<TestKernelCPU>(Dev));
  EXPECT_FALSE(sycl::is_compatible<TestKernelGPU>(Dev));
  EXPECT_TRUE(sycl::is_compatible<TestKernelACC>(Dev));
}
