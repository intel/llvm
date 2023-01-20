#include <sycl/sycl.hpp>

#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>

#include <gtest/gtest.h>

class TestKernelCPU;
class TestKernelCPUInvalidReqdWGSize1D;
class TestKernelCPUInvalidReqdWGSize2D;
class TestKernelGPU;
class TestKernelACC;

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {
template <> struct KernelInfo<TestKernelCPU> {
  static constexpr unsigned getNumParams() { return 0; }
  static const kernel_param_desc_t &getParamDesc(int) {
    static kernel_param_desc_t Dummy;
    return Dummy;
  }
  static constexpr const char *getName() { return "TestKernelCPU"; }
  static constexpr bool isESIMD() { return false; }
  static constexpr bool callsThisItem() { return false; }
  static constexpr bool callsAnyThisFreeFunction() { return false; }
  static constexpr int64_t getKernelSize() { return 1; }
};

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {
template <> struct KernelInfo<TestKernelCPUInvalidReqdWGSize1D> {
  static constexpr unsigned getNumParams() { return 0; }
  static const kernel_param_desc_t &getParamDesc(int) {
    static kernel_param_desc_t Dummy;
    return Dummy;
  }
  static constexpr const char *getName() { return "TestKernelCPUInvalidReqdWGSize1D"; }
  static constexpr bool isESIMD() { return false; }
  static constexpr bool callsThisItem() { return false; }
  static constexpr bool callsAnyThisFreeFunction() { return false; }
  static constexpr int64_t getKernelSize() { return 1; }
};

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {
template <> struct KernelInfo<TestKernelCPUInvalidReqdWGSize2D> {
  static constexpr unsigned getNumParams() { return 0; }
  static const kernel_param_desc_t &getParamDesc(int) {
    static kernel_param_desc_t Dummy;
    return Dummy;
  }
  static constexpr const char *getName() { return "TestKernelCPUInvalidReqdWGSize2D"; }
  static constexpr bool isESIMD() { return false; }
  static constexpr bool callsThisItem() { return false; }
  static constexpr bool callsAnyThisFreeFunction() { return false; }
  static constexpr int64_t getKernelSize() { return 1; }
};

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {
template <> struct KernelInfo<TestKernelGPU> {
  static constexpr unsigned getNumParams() { return 0; }
  static const kernel_param_desc_t &getParamDesc(int) {
    static kernel_param_desc_t Dummy;
    return Dummy;
  }
  static constexpr const char *getName() { return "TestKernelGPU"; }
  static constexpr bool isESIMD() { return false; }
  static constexpr bool callsThisItem() { return false; }
  static constexpr bool callsAnyThisFreeFunction() { return false; }
  static constexpr int64_t getKernelSize() { return 1; }
};

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {
template <> struct KernelInfo<TestKernelACC> {
  static constexpr unsigned getNumParams() { return 0; }
  static const kernel_param_desc_t &getParamDesc(int) {
    static kernel_param_desc_t Dummy;
    return Dummy;
  }
  static constexpr const char *getName() { return "TestKernelACC"; }
  static constexpr bool isESIMD() { return false; }
  static constexpr bool callsThisItem() { return false; }
  static constexpr bool callsAnyThisFreeFunction() { return false; }
  static constexpr int64_t getKernelSize() { return 1; }
};

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
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

static sycl::unittest::PiImage Imgs[5] = {
    generateDefaultImage({"TestKernelCPU"}, {sycl::aspect::cpu}, {32}),
    generateDefaultImage({"TestKernelCPUInvalidReqdWGSize1D"},
                         {sycl::aspect::cpu}, {257}), // 257 > 256
    generateDefaultImage({"TestKernelCPUInvalidReqdWGSize2D"},
                         {sycl::aspect::cpu}, {32, 9}), // 32*9=288 > 256
    generateDefaultImage({"TestKernelGPU"}, {sycl::aspect::gpu}),
    generateDefaultImage({"TestKernelACC"}, {sycl::aspect::accelerator})};

static sycl::unittest::PiImageArray<5> ImgArray{Imgs};

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
    auto *Result = reinterpret_cast<size_t *>(param_value);
    *Result = 256;
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
