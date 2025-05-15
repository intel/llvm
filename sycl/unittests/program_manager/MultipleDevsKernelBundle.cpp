//==----------------------- MultipleDevsKernelBundle.cpp -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Kernel bundle for multiple devices unit test

#include "detail/context_impl.hpp"
#include "detail/kernel_bundle_impl.hpp"
#include "detail/persistent_device_code_cache.hpp"
#include <detail/config.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/ScopedEnvVar.hpp>
#include <helpers/UrMock.hpp>
#include <llvm/Support/FileSystem.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <fstream>

using namespace sycl;

class MultipleDevsKernelBundleTestKernel;
class DevLibTestKernel;

MOCK_INTEGRATION_HEADER(MultipleDevsKernelBundleTestKernel)
MOCK_INTEGRATION_HEADER(DevLibTestKernel)

using namespace sycl::unittest;

inline void createDummyDeviceLib(sycl::detail::DeviceLibExt Ext) {
  // Create a dummy fallback library correpsonding to the extension (if it
  // doesn't exist).
  std::string ExtName;
  switch (Ext) {
  case sycl::detail::DeviceLibExt::cl_intel_devicelib_math:
    ExtName = "libsycl-fallback-cmath";
    break;
  case sycl::detail::DeviceLibExt::cl_intel_devicelib_assert:
    ExtName = "libsycl-fallback-cassert";
    break;
  default:
    FAIL() << "Unknown device library extension";
  }

  auto DSOPath = sycl::detail::OSUtil::getCurrentDSODir();
  std::string LibPath = DSOPath + detail::OSUtil::DirSep + ExtName + ".spv";
  std::ifstream LibFile(LibPath);
  if (LibFile.good()) {
    LibFile.close();
  } else {
    std::ofstream LibFile(LibPath);
    LibFile << "0";
    LibFile.close();
  }
}

// Function to geneate mock device image which uses device libraries.
inline sycl::unittest::MockDeviceImage generateImage(
    std::initializer_list<std::string> KernelNames,
    sycl::detail::ur::DeviceBinaryType BinType, const char *DeviceTargetSpec,
    const std::vector<sycl::detail::DeviceLibExt> &DeviceLibExts = {}) {
  // Create dummy device libraries if they don't exist.
  for (auto Ext : DeviceLibExts) {
    createDummyDeviceLib(Ext);
  }

  MockPropertySet PropSet(DeviceLibExts);

  std::string Combined;
  for (auto it = KernelNames.begin(); it != KernelNames.end(); ++it) {
    if (it != KernelNames.begin())
      Combined += ", ";
    Combined += *it;
  }
  std::vector<unsigned char> Bin(Combined.begin(), Combined.end());
  Bin.push_back(0);

  std::vector<MockOffloadEntry> Entries = makeEmptyKernels(KernelNames);

  sycl::unittest::MockDeviceImage Img{BinType,          // Format
                                      DeviceTargetSpec, // DeviceTargetSpec
                                      "",               // Compile options
                                      "",               // Link options
                                      std::move(Bin),
                                      std::move(Entries),
                                      std::move(PropSet)};
  return Img;
}

// Set of mock device images which will be used in the tests.
static sycl::unittest::MockDeviceImage Imgs[3] = {
    sycl::unittest::generateDefaultImage(
        {"MultipleDevsKernelBundleTestKernel"}),
    generateImage({"DevLibTestKernel"}, SYCL_DEVICE_BINARY_TYPE_SPIRV,
                  __SYCL_DEVICE_BINARY_TARGET_SPIRV64,
                  {sycl::detail::DeviceLibExt::cl_intel_devicelib_math,
                   sycl::detail::DeviceLibExt::cl_intel_devicelib_assert}),
    generateImage({"DevLibTestKernel"}, SYCL_DEVICE_BINARY_TYPE_NATIVE,
                  __SYCL_DEVICE_BINARY_TARGET_SPIRV64_X86_64,
                  {sycl::detail::DeviceLibExt::cl_intel_devicelib_math,
                   sycl::detail::DeviceLibExt::cl_intel_devicelib_assert})};

static sycl::unittest::MockDeviceImageArray<3> ImgArray{Imgs};

struct MockDeviceData {
  int Index;
  ur_device_type_t DeviceType;
  ur_device_handle_t getHandle() {
    return reinterpret_cast<ur_device_handle_t>(this);
  }
  static MockDeviceData *fromHandle(ur_device_handle_t handle) {
    return reinterpret_cast<MockDeviceData *>(handle);
  }
};

// List of devices.
MockDeviceData MockGPUDevices[] = {{0, UR_DEVICE_TYPE_GPU},
                                   {1, UR_DEVICE_TYPE_GPU},
                                   {2, UR_DEVICE_TYPE_GPU},
                                   {3, UR_DEVICE_TYPE_GPU}};
MockDeviceData MockCPUDevices[] = {{0, UR_DEVICE_TYPE_CPU},
                                   {1, UR_DEVICE_TYPE_CPU},
                                   {2, UR_DEVICE_TYPE_CPU},
                                   {3, UR_DEVICE_TYPE_CPU}};

static ur_result_t redefinedDeviceGet(void *pParams) {
  auto params = *static_cast<ur_device_get_params_t *>(pParams);
  size_t Size = (*params.pDeviceType == UR_DEVICE_TYPE_GPU)
                    ? std::size(MockGPUDevices)
                    : std::size(MockCPUDevices);
  MockDeviceData *MockDevices = (*params.pDeviceType == UR_DEVICE_TYPE_GPU)
                                    ? MockGPUDevices
                                    : MockCPUDevices;

  if (*params.ppNumDevices) {
    **params.ppNumDevices = static_cast<uint32_t>(Size);
    return UR_RESULT_SUCCESS;
  }

  if (*params.pphDevices) {
    assert(*params.pNumEntries <= Size);
    for (uint32_t i = 0; i < *params.pNumEntries; ++i) {
      (*params.pphDevices)[i] = MockDevices[i].getHandle();
    }
  }

  return UR_RESULT_SUCCESS;
}

// Choose SPIRV image for gpu device and Native image for cpu device.
static ur_result_t redefinedDeviceSelectBinary(void *pParams) {
  auto params = *static_cast<ur_device_select_binary_params_t *>(pParams);
  auto target = MockDeviceData::fromHandle(*params.phDevice)->DeviceType ==
                        UR_DEVICE_TYPE_CPU
                    ? UR_DEVICE_BINARY_TARGET_SPIRV64_X86_64
                    : UR_DEVICE_BINARY_TARGET_SPIRV64;
  // If compatible binary is found, select it, otherwise return -1 as an index -
  // this is what program manager expects.
  **params.ppSelectedBinary = int32_t(-1);
  for (uint32_t i = 0; i < *params.pNumBinaries; ++i) {
    if (strcmp((*params.ppBinaries)[i].pDeviceTargetSpec, target) == 0) {
      **params.ppSelectedBinary = i;
      return UR_RESULT_SUCCESS;
    }
  }
  return UR_RESULT_SUCCESS;
}

inline ur_result_t redefinedurKernelGetInfo(void *pParams) {
  auto params = *static_cast<ur_kernel_get_info_params_t *>(pParams);
  constexpr char MockKernel[] = "MultipleDevsKernelBundleTestKernel";
  if (*params.ppropName == UR_KERNEL_INFO_FUNCTION_NAME) {
    if (*params.ppPropValue) {
      assert(*params.ppropSize == sizeof(MockKernel));
      std::memcpy(*params.ppPropValue, MockKernel, sizeof(MockKernel));
    }
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(MockKernel);
  }
  return UR_RESULT_SUCCESS;
}

static ur_result_t redefinedDeviceGetInfo(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);
  if (*params.ppropName == UR_DEVICE_INFO_MULTI_DEVICE_COMPILE_SUPPORT_EXP) {
    auto *Result = reinterpret_cast<ur_bool_t *>(*params.ppPropValue);
    *Result = true;
  }
  return UR_RESULT_SUCCESS;
}

static int ProgramBuildExpCounter = 0;
static ur_result_t redefinedurProgramBuildExp(void *) {
  ++ProgramBuildExpCounter;
  return UR_RESULT_SUCCESS;
}

static int ProgramCreateWithILCounter = 0;
static ur_result_t redefinedurProgramCreateWithIL(void *) {
  ++ProgramCreateWithILCounter;
  return UR_RESULT_SUCCESS;
}

static int ProgramLinkExpCounter = 0;
static ur_result_t redefinedurProgramLinkExp(void *) {
  ++ProgramLinkExpCounter;
  return UR_RESULT_SUCCESS;
}

static int ProgramCompileExpCounter = 0;
static ur_result_t redefinedurProgramCompileExp(void *) {
  ++ProgramCompileExpCounter;
  return UR_RESULT_SUCCESS;
}

static int ProgramCreateWithBinaryCounter = 0;
static ur_result_t redefinedurProgramCreateWithBinary(void *) {
  ++ProgramCreateWithBinaryCounter;
  return UR_RESULT_SUCCESS;
}

class MultipleDevsKernelBundleTest
    : public testing::TestWithParam<sycl::detail::ur::DeviceBinaryType> {
public:
  MultipleDevsKernelBundleTest() : Mock{}, Plt{sycl::platform()} {}

protected:
  void SetUp() override {
    mock::getCallbacks().set_after_callback("urDeviceGet", &redefinedDeviceGet);
    mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                            &redefinedDeviceGetInfo);
    mock::getCallbacks().set_after_callback("urKernelGetInfo",
                                            &redefinedurKernelGetInfo);
    mock::getCallbacks().set_after_callback("urProgramBuildExp",
                                            &redefinedurProgramBuildExp);
    mock::getCallbacks().set_after_callback("urProgramCreateWithIL",
                                            &redefinedurProgramCreateWithIL);
    mock::getCallbacks().set_after_callback("urProgramLinkExp",
                                            &redefinedurProgramLinkExp);
    mock::getCallbacks().set_after_callback("urProgramCompileExp",
                                            &redefinedurProgramCompileExp);
    mock::getCallbacks().set_after_callback("urDeviceSelectBinary",
                                            &redefinedDeviceSelectBinary);
    mock::getCallbacks().set_after_callback(
        "urProgramCreateWithBinary", &redefinedurProgramCreateWithBinary);
  }

protected:
  unittest::UrMock<> Mock;
  platform Plt;
};

// Test to check that we can create input kernel bundle and call build twice for
// overlapping set of devices and execute the kernel on each device.
TEST_P(MultipleDevsKernelBundleTest, BuildTwiceWithOverlappingDevices) {
  // Reset counters
  ProgramCreateWithILCounter = 0;
  ProgramBuildExpCounter = 0;

  // Get devices and create a context with at least 3 devices
  std::vector<sycl::device> Devices =
      Plt.get_devices(sycl::info::device_type::gpu);
  ASSERT_GE(Devices.size(), 3lu) << "Test requires at least 3 devices";

  auto Dev1 = Devices[0], Dev2 = Devices[1], Dev3 = Devices[2];

  // Create a context with the selected devices
  sycl::context Context({Dev1, Dev2, Dev3});

  // Create queues for each device
  sycl::queue Queue1(Context, Dev1);
  sycl::queue Queue2(Context, Dev2);
  sycl::queue Queue3(Context, Dev3);

  // Get kernel ID
  auto KernelID = sycl::get_kernel_id<MultipleDevsKernelBundleTestKernel>();

  // Create an input kernel bundle
  auto KernelBundleInput =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Context, {KernelID});

  // Build kernel bundles for overlapping sets of devices
  auto KernelBundleExe1 = sycl::build(KernelBundleInput, {Dev1, Dev2});
  auto KernelBundleExe2 = sycl::build(KernelBundleInput, {Dev2, Dev3});

  // Get kernel objects from the built bundles
  auto KernelObj1 = KernelBundleExe1.get_kernel(KernelID);
  auto KernelObj2 = KernelBundleExe2.get_kernel(KernelID);

  // Submit tasks to the queues using the kernel bundles
  Queue1.submit([&](sycl::handler &cgh) {
    cgh.use_kernel_bundle(KernelBundleExe1);
    cgh.single_task<MultipleDevsKernelBundleTestKernel>([]() {});
  });

  Queue2.submit([&](sycl::handler &cgh) {
    cgh.use_kernel_bundle(KernelBundleExe1);
    cgh.single_task(KernelObj1);
  });

  Queue2.submit([&](sycl::handler &cgh) {
    cgh.use_kernel_bundle(KernelBundleExe2);
    cgh.single_task(KernelObj2);
  });

  Queue3.submit([&](sycl::handler &cgh) {
    cgh.use_kernel_bundle(KernelBundleExe2);
    cgh.single_task(KernelObj2);
  });

  // Verify the number of urProgramCreateWithIL calls
  EXPECT_EQ(ProgramCreateWithILCounter, 2)
      << "Expect 2 urProgramCreateWithIL calls";

  // Verify the number of urProgramBuildExp calls
  EXPECT_EQ(ProgramBuildExpCounter, 2) << "Expect 2 urProgramBuildExp calls";
}

// Test to check several use cases for multi-device kernel bundles.
// Test covers AOT and JIT cases. We mock usage of fallback device libaries to
// excersise additional logic in the program manager. Checks are used to test
// that program and device libraries caching works as expected.
TEST_P(MultipleDevsKernelBundleTest, DeviceLibs) {
  // Unset the SYCL_DEVICELIB_NO_FALLBACK so that fallback libraries are used.
  ScopedEnvVar var("SYCL_DEVICELIB_NO_FALLBACK", nullptr,
                   SYCLConfig<SYCL_DEVICELIB_NO_FALLBACK>::reset);
  std::vector<sycl::device> Devices =
      Plt.get_devices(GetParam() == SYCL_DEVICE_BINARY_TYPE_NATIVE
                          ? sycl::info::device_type::cpu
                          : sycl::info::device_type::gpu);
  ASSERT_GE(Devices.size(), 4lu) << "Test requires at least 4 devices";

  auto Dev1 = Devices[0], Dev2 = Devices[1], Dev3 = Devices[2],
       Dev4 = Devices[3];

  // Create a context with the selected devices
  sycl::context Context({Dev1, Dev2, Dev3, Dev4});
  sycl::queue Queues[4] = {
      sycl::queue(Context, Dev1), sycl::queue(Context, Dev2),
      sycl::queue(Context, Dev3), sycl::queue(Context, Dev4)};
  {
    // Test case 1
    // Get bundle in executable state for multiple devices in a context, enqueue
    // a kernel to each device.

    // Reset counters
    ProgramCreateWithILCounter = 0;
    ProgramBuildExpCounter = 0;
    ProgramLinkExpCounter = 0;
    ProgramCompileExpCounter = 0;
    ProgramCreateWithBinaryCounter = 0;

    // Get bundle in executable state for multiple devices in a context, enqueue
    // a kernel to each device.
    sycl::kernel_id KernelID = sycl::get_kernel_id<DevLibTestKernel>();
    sycl::kernel_bundle KernelBundleExecutable =
        sycl::get_kernel_bundle<sycl::bundle_state::executable>(
            Context, {Dev1, Dev2, Dev3}, {KernelID});
    for (int i = 0; i < 2; i++) {
      Queues[i].submit([=](sycl::handler &cgh) {
        cgh.use_kernel_bundle(KernelBundleExecutable);
        cgh.single_task<DevLibTestKernel>([=]() {});
      });
      Queues[i].wait();
    }

    if (GetParam() == SYCL_DEVICE_BINARY_TYPE_SPIRV) {
      // Verify the number of urProgramCreateWithIL calls: we expect 2 calls for
      // fallback libraries (assert + math) and 1 call for the main program.
      EXPECT_EQ(ProgramCreateWithILCounter, 3)
          << "Expect 3 urProgramCreateWithIL calls";

      // Verify the number of urProgramBuildExp calls: none expected as we
      // compile and link in this case.
      EXPECT_EQ(ProgramBuildExpCounter, 0)
          << "Expect 0 urProgramBuildExp calls";

      // Verify the number of urProgramCompileExp calls: we expect 2 calls to
      // compile fallback libraries and 1 call to compile the main program.
      EXPECT_EQ(ProgramCompileExpCounter, 3)
          << "Expect 3 urProgramCompileExp calls";

      // Verify the number of urProgramLinkExp calls: we expect 1 call which
      // links the main program and fallback libraries.
      EXPECT_EQ(ProgramLinkExpCounter, 1) << "Expect 1 urProgramLinkExp calls";
    }
    if (GetParam() == SYCL_DEVICE_BINARY_TYPE_NATIVE) {
      // In case of AOT compilation, we expect 1 call to
      // urProgramCreateWithBinary.
      EXPECT_EQ(ProgramCreateWithBinaryCounter, 1)
          << "Expect 3 urProgramCreateWithIL calls";

      // And a single call to urProgramBuildExp. In this case libraries are
      // linked beforehand, so we don't compile/link them online.
      EXPECT_EQ(ProgramBuildExpCounter, 1)
          << "Expect 0 urProgramBuildExp calls";
    }
  }

  {

    // Test case 2
    // Get bundles in executable state: for pairs of devices excluding dev4 and
    // for the new set of devices which includes the dev4. This checks caching
    // of the programs and device libraries.

    // Reset counters
    ProgramCreateWithILCounter = 0;
    ProgramBuildExpCounter = 0;
    ProgramLinkExpCounter = 0;
    ProgramCompileExpCounter = 0;
    ProgramCreateWithBinaryCounter = 0;
    sycl::kernel_id KernelID = sycl::get_kernel_id<DevLibTestKernel>();
    // Program associated with {dev1, dev2, dev3} is supposed to be cached from
    // the first test case, we don't expect any additional program creation and
    // compilation calls for the following bundles because they are all created
    // for subsets of {dev1, dev2, dev3} which means that the program handle
    // from cache will be used.
    sycl::kernel_bundle KernelBundleExecutableSubset1 =
        sycl::get_kernel_bundle<sycl::bundle_state::executable>(
            Context, {Dev1, Dev2}, {KernelID});
    sycl::kernel_bundle KernelBundleExecutableSubset2 =
        sycl::get_kernel_bundle<sycl::bundle_state::executable>(
            Context, {Dev2, Dev3}, {KernelID});
    sycl::kernel_bundle KernelBundleExecutableSubset3 =
        sycl::get_kernel_bundle<sycl::bundle_state::executable>(
            Context, {Dev1, Dev3}, {KernelID});
    sycl::kernel_bundle KernelBundleExecutableSubset4 =
        sycl::get_kernel_bundle<sycl::bundle_state::executable>(Context, {Dev3},
                                                                {KernelID});
    EXPECT_EQ(ProgramCreateWithILCounter, 0);
    EXPECT_EQ(ProgramCompileExpCounter, 0);
    EXPECT_EQ(ProgramLinkExpCounter, 0);

    // Next we create a bundle with a different set of devices which includes
    // dev4, so we expect new UR program creation. Also main program will be
    // compiled for new set of devices. Each of device libraries (assert and
    // math) will be additionally compiled for dev4, but no program creation is
    // expected for device libraries as program handle already exists in the
    // per-context cache.
    sycl::kernel_bundle KernelBundleExecutableNewSet =
        sycl::get_kernel_bundle<sycl::bundle_state::executable>(
            Context, {Dev2, Dev3, Dev4}, {KernelID});
    if (GetParam() == SYCL_DEVICE_BINARY_TYPE_SPIRV) {
      EXPECT_EQ(ProgramCreateWithILCounter, 1)
          << "Expect 1 urProgramCreateWithIL calls";
      EXPECT_EQ(ProgramCompileExpCounter, 3)
          << "Expect 3 urProgramCompileExp calls";
      EXPECT_EQ(ProgramLinkExpCounter, 1) << "Expect 1 urProgramLinkExp calls";
    }

    if (GetParam() == SYCL_DEVICE_BINARY_TYPE_NATIVE) {
      EXPECT_EQ(ProgramCreateWithBinaryCounter, 1)
          << "Expect 1 urProgramCreateWithBinary calls";
      EXPECT_EQ(ProgramBuildExpCounter, 1)
          << "Expect 1 urProgramBuildExp calls";
    }

    for (int i = 0; i < 3; i++) {
      Queues[0].submit([=](sycl::handler &cgh) {
        cgh.use_kernel_bundle(KernelBundleExecutableSubset1);
        cgh.single_task<DevLibTestKernel>([=]() {});
      });
      Queues[0].wait();

      Queues[2].submit([=](sycl::handler &cgh) {
        cgh.use_kernel_bundle(KernelBundleExecutableNewSet);
        cgh.single_task<DevLibTestKernel>([=]() {});
      });
      Queues[2].wait();
    }
  }

  // Reset the SYCL_DEVICELIB_NO_FALLBACK to its original value.
  sycl::detail::SYCLConfig<sycl::detail::SYCL_DEVICELIB_NO_FALLBACK>::reset();
}

// The following helpers and test verify persistent cache usage when we have
// kernel bundle with multiple devices.
#define ASSERT_NO_ERROR(x)                                                     \
  if (std::error_code EC = x) {                                                \
    FAIL() << #x ": did not return errc::success.\n"                           \
           << "error number: " << EC.value() << "\n"                           \
           << "error message: " << EC.message() << "\n";                       \
  }

std::vector<int> Prog = {125, 1024, 256, 32};

static ur_result_t redefinedurProgramGetInfo(void *pParams) {
  auto params = *static_cast<ur_program_get_info_params_t *>(pParams);
  if (*params.ppropName == UR_PROGRAM_INFO_NUM_DEVICES) {
    auto value = reinterpret_cast<unsigned int *>(*params.ppPropValue);
    *value = Prog.size();
  }

  if (*params.ppropName == UR_PROGRAM_INFO_DEVICES) {
    if (*params.ppPropValue) {
      for (size_t i = 0; i < Prog.size(); i++) {
        auto devs = static_cast<ur_device_handle_t *>(*params.ppPropValue);
        devs[i] = MockGPUDevices[i].getHandle();
      }
    }
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(ur_device_handle_t) * Prog.size();
    return UR_RESULT_SUCCESS;
  }

  if (*params.ppropName == UR_PROGRAM_INFO_BINARY_SIZES) {
    auto value = reinterpret_cast<size_t *>(*params.ppPropValue);
    for (size_t i = 0; i < Prog.size(); ++i)
      value[i] = Prog[i];
  }

  if (*params.ppropName == UR_PROGRAM_INFO_BINARIES) {
    auto value = reinterpret_cast<unsigned char **>(*params.ppPropValue);
    for (size_t i = 0; i < Prog.size(); ++i) {
      for (int j = 0; j < Prog[i]; ++j) {
        value[i][j] = i;
      }
    }
  }

  return UR_RESULT_SUCCESS;
}

#if defined(__linux__)
extern char **environ;
#endif

// https://github.com/intel/llvm/issues/18122
TEST_P(MultipleDevsKernelBundleTest, PersistentCache) {
  // Create temporary directory for the persistent cache in the directory of the
  // test binary.
  std::string PersistentCachePath = sycl::detail::OSUtil::getCurrentDSODir() +
                                    detail::OSUtil::DirSep + "persistent_cache";
  // Set environment variables to enable persistent cache and set the cache
  // path.
  ScopedEnvVar var1("SYCL_CACHE_PERSISTENT", "1",
                    SYCLConfig<SYCL_CACHE_PERSISTENT>::reset);
  ScopedEnvVar var2("SYCL_CACHE_DIR", PersistentCachePath.c_str(),
                    SYCLConfig<SYCL_CACHE_DIR>::reset);
  ScopedEnvVar cach_trace_var("SYCL_CACHE_TRACE", "1",
                   SYCLConfig<SYCL_DEVICELIB_NO_FALLBACK>::reset);

#if defined(__linux__)
    for (char **env = environ; *env != nullptr; ++env) {
        std::cout << *env << std::endl;
    }
#endif
  // Disable in-memory cache in this test case, as we are interested in
  // persistent cache usage.
  ScopedEnvVar var3("SYCL_CACHE_IN_MEM", "0",
                    SYCLConfig<SYCL_CACHE_IN_MEM>::reset);

  mock::getCallbacks().set_after_callback("urProgramGetInfo",
                                          &redefinedurProgramGetInfo);
  std::string CacheRoot = detail::PersistentDeviceCodeCache::getRootDir();
  ASSERT_NO_ERROR(llvm::sys::fs::remove_directories(CacheRoot));
  ASSERT_NO_ERROR(llvm::sys::fs::create_directories(CacheRoot));

  // Get devices and create a context with at least 3 devices
  std::vector<sycl::device> Devices =
      Plt.get_devices(sycl::info::device_type::gpu);
  ASSERT_GE(Devices.size(), 3lu) << "Test requires at least 3 devices";

  // Create a context with the selected devices
  sycl::context Context({Devices[0], Devices[1], Devices[2], Devices[3]});
  auto KernelID = sycl::get_kernel_id<MultipleDevsKernelBundleTestKernel>();
  auto Bundle =
      sycl::get_kernel_bundle<bundle_state::input>(Context, {KernelID});

  auto BundleExe = sycl::build(Bundle, {Devices[0], Devices[2]});

  // Verify that binaries that we get from build stage for each device are put
  // into the persistent cache.
  sycl_device_binary_struct BinStruct = Imgs[0].convertToNativeType();
  sycl_device_binary Bin = &BinStruct;
  detail::RTDeviceBinaryImage RTBinImg{Bin};
  auto Res = detail::PersistentDeviceCodeCache::getItemFromDisc(
      {Devices[0], Devices[2]}, {&RTBinImg}, {}, {});
  EXPECT_EQ(Res.size(), static_cast<size_t>(2))
      << "Expected cache items to be loaded";

  // Now check that binaries from persistent cache are used to create a program
  // when we submit a kernel.
  ProgramCreateWithBinaryCounter = 0;

  sycl::queue q(Devices[2]);
  q.submit([&](sycl::handler &cgh) {
    cgh.single_task<MultipleDevsKernelBundleTestKernel>([=]() {});
  });
  q.wait();

  // Verify the number of urProgramCreateWithBinary calls
  EXPECT_EQ(ProgramCreateWithBinaryCounter, 1)
      << "Expect 1 urProgramCreateWithBinary calls";
  ASSERT_NO_ERROR(llvm::sys::fs::remove_directories(CacheRoot));
}

INSTANTIATE_TEST_SUITE_P(MultipleDevsKernelBundleTestInstance,
                         MultipleDevsKernelBundleTest,
                         testing::Values(SYCL_DEVICE_BINARY_TYPE_SPIRV,
                                         SYCL_DEVICE_BINARY_TYPE_NATIVE));
