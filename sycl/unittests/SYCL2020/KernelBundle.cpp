//==---- DefaultValues.cpp --- Spec constants default values unit test -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/device_impl.hpp>
#include <detail/kernel_bundle_impl.hpp>
#include <sycl/sycl.hpp>

#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

class TestKernel;
class TestKernelExeOnly;
class TestKernelWithAspects;

MOCK_INTEGRATION_HEADER(TestKernel)
MOCK_INTEGRATION_HEADER(TestKernelExeOnly)
MOCK_INTEGRATION_HEADER(TestKernelWithAspects)

static sycl::unittest::MockDeviceImage
generateDefaultImage(std::initializer_list<std::string> KernelNames,
                     sycl_device_binary_type BinaryType,
                     const char *DeviceTargetSpec,
                     const std::vector<sycl::aspect> &Aspects = {}) {
  using namespace sycl::unittest;

  MockPropertySet PropSet;
  if (!Aspects.empty())
    addDeviceRequirementsProps(PropSet, Aspects);

  std::vector<unsigned char> Bin{0, 1, 2, 3, 4, 5}; // Random data

  std::vector<MockOffloadEntry> Entries = makeEmptyKernels(KernelNames);

  MockDeviceImage Img{BinaryType, // Format
                      DeviceTargetSpec,
                      "", // Compile options
                      "", // Link options
                      std::move(Bin),
                      std::move(Entries),
                      std::move(PropSet)};

  return Img;
}

static sycl::unittest::MockDeviceImage Imgs[] = {
    generateDefaultImage({"TestKernel"}, SYCL_DEVICE_BINARY_TYPE_SPIRV,
                         __SYCL_DEVICE_BINARY_TARGET_SPIRV64),
    generateDefaultImage({"TestKernelExeOnly"}, SYCL_DEVICE_BINARY_TYPE_NATIVE,
                         __SYCL_DEVICE_BINARY_TARGET_SPIRV64_X86_64),
    // A device image without entires
    generateDefaultImage({}, SYCL_DEVICE_BINARY_TYPE_NATIVE,
                         __SYCL_DEVICE_BINARY_TARGET_SPIRV64_X86_64),
    generateDefaultImage(
        {"TestKernelWithAspects"}, SYCL_DEVICE_BINARY_TYPE_NATIVE,
        __SYCL_DEVICE_BINARY_TARGET_SPIRV64, {sycl::aspect::gpu})};
static sycl::unittest::MockDeviceImageArray<std::size(Imgs)> ImgArray{Imgs};

static ur_result_t redefinedDeviceGetInfoCPU(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);
  if (*params.ppropName == UR_DEVICE_INFO_TYPE) {
    auto *Result = reinterpret_cast<ur_device_type_t *>(*params.ppPropValue);
    *Result = UR_DEVICE_TYPE_CPU;
  }
  return UR_RESULT_SUCCESS;
}

TEST(KernelBundle, GetKernelBundleFromKernel) {
  sycl::unittest::UrMock<> Mock;

  const sycl::device Dev = sycl::platform().get_devices()[0];
  sycl::context Ctx{Dev};
  sycl::queue Queue{Ctx, Dev};

  sycl::kernel_bundle<sycl::bundle_state::executable> KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(Ctx, {Dev});

  sycl::kernel Kernel =
      KernelBundle.get_kernel(sycl::get_kernel_id<TestKernel>());

  sycl::kernel_bundle<sycl::bundle_state::executable> RetKernelBundle =
      Kernel.get_kernel_bundle();

  EXPECT_EQ(KernelBundle, RetKernelBundle);
}

TEST(KernelBundle, KernelBundleAndItsDevImageStateConsistency) {
  sycl::unittest::UrMock<> Mock;

  const sycl::device Dev = sycl::platform().get_devices()[0];

  sycl::queue Queue{Dev};

  const sycl::context Ctx = Queue.get_context();

  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<TestKernel, sycl::bundle_state::input>(Ctx,
                                                                     {Dev});

  auto ObjBundle = sycl::compile(KernelBundle, KernelBundle.get_devices());
  EXPECT_FALSE(ObjBundle.empty()) << "Expect non-empty obj kernel bundle";

  auto ObjBundleImpl = sycl::detail::getSyclObjImpl(ObjBundle);
  EXPECT_EQ(ObjBundleImpl->get_bundle_state(), sycl::bundle_state::object)
      << "Expect object device image in bundle";

  auto LinkBundle = sycl::link(ObjBundle, ObjBundle.get_devices());
  EXPECT_FALSE(LinkBundle.empty()) << "Expect non-empty exec kernel bundle";

  auto LinkBundleImpl = sycl::detail::getSyclObjImpl(LinkBundle);
  EXPECT_EQ(LinkBundleImpl->get_bundle_state(), sycl::bundle_state::executable)
      << "Expect executable device image in bundle";
}

TEST(KernelBundle, EmptyKernelBundle) {
  sycl::unittest::UrMock<> Mock;

  const sycl::device Dev = sycl::platform().get_devices()[0];
  sycl::queue Queue{Dev};
  const sycl::context Ctx = Queue.get_context();

  auto EmptyKernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(Ctx, {Dev}, {});

  EXPECT_TRUE(EmptyKernelBundle.empty());
  EXPECT_EQ(std::distance(EmptyKernelBundle.begin(), EmptyKernelBundle.end()),
            0u);
}

TEST(KernelBundle, EmptyKernelBundleKernelLaunchException) {
  sycl::unittest::UrMock<> Mock;

  const sycl::device Dev = sycl::platform().get_devices()[0];
  sycl::queue Queue{Dev};
  const sycl::context Ctx = Queue.get_context();

  auto EmptyKernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(Ctx, {Dev}, {});

  class UnqiueException {};

  try {
    Queue.submit([&](sycl::handler &CGH) {
      CGH.use_kernel_bundle(EmptyKernelBundle);

      try {
        CGH.single_task<TestKernel>([]() {});
        FAIL() << "No exception was thrown.";
      } catch (const sycl::exception &e) {
        ASSERT_EQ(e.code().value(),
                  static_cast<int>(sycl::errc::kernel_not_supported))
            << "sycl::exception code was not the expected "
               "sycl::errc::kernel_not_supported.";
        // Throw uniquely identifiable exception to distinguish between that
        // the sycl::exception originates from the correct level.
        throw UnqiueException{};
      } catch (...) {
        FAIL()
            << "Unexpected exception was thrown in kernel invocation function.";
      }
    });
  } catch (const UnqiueException &) {
    // Expected path
  } catch (const sycl::exception &) {
    FAIL() << "sycl::exception thrown at the wrong level.";
  } catch (...) {
    FAIL() << "Unexpected exception was thrown in submit.";
  }
}

TEST(KernelBundle, HasKernelBundle) {
  sycl::unittest::UrMock<> Mock;

  const sycl::device Dev = sycl::platform().get_devices()[0];

  sycl::queue Queue{Dev};

  const sycl::context Ctx = Queue.get_context();

  bool HasKernelBundle =
      sycl::has_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev});
  EXPECT_TRUE(HasKernelBundle);
  HasKernelBundle =
      sycl::has_kernel_bundle<sycl::bundle_state::object>(Ctx, {Dev});
  EXPECT_TRUE(HasKernelBundle);
  HasKernelBundle =
      sycl::has_kernel_bundle<sycl::bundle_state::executable>(Ctx, {Dev});
  EXPECT_TRUE(HasKernelBundle);

  HasKernelBundle =
      sycl::has_kernel_bundle<TestKernel, sycl::bundle_state::input>(Ctx,
                                                                     {Dev});
  EXPECT_TRUE(HasKernelBundle);
  HasKernelBundle =
      sycl::has_kernel_bundle<TestKernel, sycl::bundle_state::object>(Ctx,
                                                                      {Dev});
  EXPECT_TRUE(HasKernelBundle);
  HasKernelBundle =
      sycl::has_kernel_bundle<TestKernel, sycl::bundle_state::executable>(
          Ctx, {Dev});
  EXPECT_TRUE(HasKernelBundle);

  HasKernelBundle =
      sycl::has_kernel_bundle<TestKernelExeOnly, sycl::bundle_state::input>(
          Ctx, {Dev});
  EXPECT_FALSE(HasKernelBundle);
  HasKernelBundle =
      sycl::has_kernel_bundle<TestKernelExeOnly, sycl::bundle_state::object>(
          Ctx, {Dev});
  EXPECT_FALSE(HasKernelBundle);
  HasKernelBundle =
      sycl::has_kernel_bundle<TestKernelExeOnly,
                              sycl::bundle_state::executable>(Ctx, {Dev});
  EXPECT_TRUE(HasKernelBundle);
}

TEST(KernelBundle, UseKernelBundleWrongContextPrimaryQueueOnly) {
  sycl::unittest::UrMock<> Mock;

  const sycl::device Dev = sycl::platform().get_devices()[0];
  sycl::queue Queue{Dev};
  const sycl::context QueueCtx = Queue.get_context();
  const sycl::context OtherCtx{Dev};

  ASSERT_NE(QueueCtx, OtherCtx);

  auto KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(OtherCtx, {Dev});

  class UnqiueException {};

  try {
    Queue.submit([&](sycl::handler &CGH) {
      try {
        CGH.use_kernel_bundle(KernelBundle);
        FAIL() << "No exception was thrown.";
        CGH.single_task<TestKernel>([]() {});
      } catch (const sycl::exception &e) {
        ASSERT_EQ(e.code().value(), static_cast<int>(sycl::errc::invalid))
            << "sycl::exception code was not the expected sycl::errc::invalid.";
        // Throw uniquely identifiable exception to distinguish between that
        // the sycl::exception originates from the correct level.
        throw UnqiueException{};
      } catch (...) {
        FAIL()
            << "Unexpected exception was thrown in kernel invocation function.";
      }
    });
  } catch (const UnqiueException &) {
    // Expected path
  } catch (const sycl::exception &) {
    FAIL() << "sycl::exception thrown at the wrong level.";
  } catch (...) {
    FAIL() << "Unexpected exception was thrown in submit.";
  }
}

TEST(KernelBundle, UseKernelBundleWrongContextPrimaryQueueValidSecondaryQueue) {
  sycl::unittest::UrMock<> Mock;

  const sycl::device Dev = sycl::platform().get_devices()[0];
  const sycl::context PrimaryCtx{Dev};
  const sycl::context SecondaryCtx{Dev};

  ASSERT_NE(PrimaryCtx, SecondaryCtx);

  auto KernelBundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      SecondaryCtx, {Dev});

  sycl::queue PrimaryQueue{PrimaryCtx, Dev};
  sycl::queue SecondaryQueue{SecondaryCtx, Dev};

  class UnqiueException {};

  try {
    PrimaryQueue.submit(
        [&](sycl::handler &CGH) {
          try {
            CGH.use_kernel_bundle(KernelBundle);
            FAIL() << "No exception was thrown.";
            CGH.single_task<TestKernel>([]() {});
          } catch (const sycl::exception &e) {
            ASSERT_EQ(e.code().value(), static_cast<int>(sycl::errc::invalid))
                << "sycl::exception code was not the expected "
                   "sycl::errc::invalid.";
            // Throw uniquely identifiable exception to distinguish between that
            // the sycl::exception originates from the correct level.
            throw UnqiueException{};
          } catch (...) {
            FAIL() << "Unexpected exception was thrown in kernel invocation "
                      "function.";
          }
        },
        SecondaryQueue);
  } catch (const UnqiueException &) {
    // Expected path
  } catch (const sycl::exception &) {
    FAIL() << "sycl::exception thrown at the wrong level.";
  } catch (...) {
    FAIL() << "Unexpected exception was thrown in submit.";
  }
}

TEST(KernelBundle, UseKernelBundleValidPrimaryQueueWrongContextSecondaryQueue) {
  sycl::unittest::UrMock<> Mock;

  const sycl::device Dev = sycl::platform().get_devices()[0];
  const sycl::context PrimaryCtx{Dev};
  const sycl::context SecondaryCtx{Dev};

  ASSERT_NE(PrimaryCtx, SecondaryCtx);

  auto KernelBundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      PrimaryCtx, {Dev});

  sycl::queue PrimaryQueue{PrimaryCtx, Dev};
  sycl::queue SecondaryQueue{SecondaryCtx, Dev};

  class UnqiueException {};

  try {
    PrimaryQueue.submit(
        [&](sycl::handler &CGH) {
          try {
            CGH.use_kernel_bundle(KernelBundle);
            FAIL() << "No exception was thrown.";
            CGH.single_task<TestKernel>([]() {});
          } catch (const sycl::exception &e) {
            ASSERT_EQ(e.code().value(), static_cast<int>(sycl::errc::invalid))
                << "sycl::exception code was not the expected "
                   "sycl::errc::invalid.";
            // Throw uniquely identifiable exception to distinguish between that
            // the sycl::exception originates from the correct level.
            throw UnqiueException{};
          } catch (...) {
            FAIL() << "Unexpected exception was thrown in kernel invocation "
                      "function.";
          }
        },
        SecondaryQueue);
  } catch (const UnqiueException &) {
    // Expected path
  } catch (const sycl::exception &) {
    FAIL() << "sycl::exception thrown at the wrong level.";
  } catch (...) {
    FAIL() << "Unexpected exception was thrown in submit.";
  }
}

TEST(KernelBundle, UseKernelBundleWrongContextPrimaryQueueAndSecondaryQueue) {
  sycl::unittest::UrMock<> Mock;

  const sycl::device Dev = sycl::platform().get_devices()[0];
  const sycl::context PrimaryCtx{Dev};
  const sycl::context SecondaryCtx{Dev};
  const sycl::context OtherCtx{Dev};

  ASSERT_NE(PrimaryCtx, SecondaryCtx);
  ASSERT_NE(PrimaryCtx, OtherCtx);
  ASSERT_NE(SecondaryCtx, OtherCtx);

  auto KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(OtherCtx, {Dev});

  sycl::queue PrimaryQueue{PrimaryCtx, Dev};
  sycl::queue SecondaryQueue{SecondaryCtx, Dev};

  class UnqiueException {};

  try {
    PrimaryQueue.submit(
        [&](sycl::handler &CGH) {
          try {
            CGH.use_kernel_bundle(KernelBundle);
            FAIL() << "No exception was thrown.";
            CGH.single_task<TestKernel>([]() {});
          } catch (const sycl::exception &e) {
            ASSERT_EQ(e.code().value(), static_cast<int>(sycl::errc::invalid))
                << "sycl::exception code was not the expected "
                   "sycl::errc::invalid.";
            // Throw uniquely identifiable exception to distinguish between that
            // the sycl::exception originates from the correct level.
            throw UnqiueException{};
          } catch (...) {
            FAIL() << "Unexpected exception was thrown in kernel invocation "
                      "function.";
          }
        },
        SecondaryQueue);
  } catch (const UnqiueException &) {
    // Expected path
  } catch (const sycl::exception &) {
    FAIL() << "sycl::exception thrown at the wrong level.";
  } catch (...) {
    FAIL() << "Unexpected exception was thrown in submit.";
  }
}

TEST(KernelBundle, EmptyDevicesKernelBundleLinkException) {
  sycl::unittest::UrMock<> Mock;

  const sycl::device Dev = sycl::platform().get_devices()[0];
  sycl::queue Queue{Dev};
  const sycl::context Ctx = Queue.get_context();

  auto EmptyKernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::object>(Ctx, {Dev}, {});

  std::vector<sycl::device> EmptyDevices{};

  // Case without an object bundle.
  try {
    auto LinkBundle = sycl::link(EmptyKernelBundle, EmptyDevices);
    FAIL() << "No exception was thrown.";
  } catch (const sycl::exception &e) {
    ASSERT_EQ(e.code().value(), static_cast<int>(sycl::errc::invalid))
        << "sycl::exception code was not the expected "
           "sycl::errc::invalid.";
    // Expected path
  } catch (...) {
    FAIL() << "Unexpected exception was thrown in sycl::link.";
  }

  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<TestKernel, sycl::bundle_state::input>(Ctx,
                                                                     {Dev});

  auto ObjBundle = sycl::compile(KernelBundle, KernelBundle.get_devices());
  EXPECT_FALSE(ObjBundle.empty()) << "Expect non-empty obj kernel bundle";

  try {
    auto LinkBundle = sycl::link(ObjBundle, EmptyDevices);
    FAIL() << "No exception was thrown.";
  } catch (const sycl::exception &e) {
    ASSERT_EQ(e.code().value(), static_cast<int>(sycl::errc::invalid))
        << "sycl::exception code was not the expected "
           "sycl::errc::invalid.";
    // Expected path
  } catch (...) {
    FAIL() << "Unexpected exception was thrown in sycl::link.";
  }
}

ur_device_handle_t ParentDevice = nullptr;
ur_platform_handle_t UrPlatform = nullptr;

ur_result_t redefinedDeviceGetInfoAfter(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);
  if (*params.ppropName == UR_DEVICE_INFO_SUPPORTED_PARTITIONS) {
    if (*params.ppPropValue) {
      auto *Result =
          reinterpret_cast<ur_device_partition_t *>(*params.ppPropValue);
      *Result = UR_DEVICE_PARTITION_EQUALLY;
    }
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(ur_device_partition_t);
  } else if (*params.ppropName == UR_DEVICE_INFO_MAX_COMPUTE_UNITS) {
    auto *Result = reinterpret_cast<uint32_t *>(*params.ppPropValue);
    *Result = 2;
  } else if (*params.ppropName == UR_DEVICE_INFO_PARENT_DEVICE) {
    auto *Result = reinterpret_cast<ur_device_handle_t *>(*params.ppPropValue);
    *Result = (*params.phDevice == ParentDevice) ? nullptr : ParentDevice;
  } else if (*params.ppropName == UR_DEVICE_INFO_PLATFORM) {
    auto *Result =
        reinterpret_cast<ur_platform_handle_t *>(*params.ppPropValue);
    *Result = UrPlatform;
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedDevicePartitionAfter(void *pParams) {
  auto params = *static_cast<ur_device_partition_params_t *>(pParams);
  if (*params.pphSubDevices) {
    for (size_t I = 0; I < *params.pNumDevices; ++I) {
      *params.pphSubDevices[I] = reinterpret_cast<ur_device_handle_t>(1000 + I);
    }
  }
  if (*params.ppNumDevicesRet)
    **params.ppNumDevicesRet = *params.pNumDevices;
  return UR_RESULT_SUCCESS;
}

TEST(KernelBundle, DescendentDevice) {
  // Mock a non-OpenCL adapter since use of descendent devices of context
  // members is not supported there yet.
  sycl::unittest::UrMock<sycl::backend::ext_oneapi_level_zero> Mock;

  sycl::platform Plt = sycl::platform();

  UrPlatform = sycl::detail::getSyclObjImpl(Plt)->getHandleRef();

  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &redefinedDeviceGetInfoAfter);
  mock::getCallbacks().set_after_callback("urDevicePartition",
                                          &redefinedDevicePartitionAfter);

  const sycl::device Dev = sycl::platform().get_devices()[0];
  ParentDevice = sycl::detail::getSyclObjImpl(Dev)->getHandleRef();
  sycl::context Ctx{Dev};
  sycl::device Subdev =
      Dev.create_sub_devices<sycl::info::partition_property::partition_equally>(
          2)[0];

  sycl::queue Queue{Ctx, Subdev};

  sycl::kernel_bundle<sycl::bundle_state::executable> KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(Ctx, {Subdev});

  sycl::kernel Kernel =
      KernelBundle.get_kernel(sycl::get_kernel_id<TestKernel>());

  sycl::kernel_bundle<sycl::bundle_state::executable> RetKernelBundle =
      Kernel.get_kernel_bundle();

  EXPECT_EQ(KernelBundle, RetKernelBundle);
}

TEST(KernelBundle, CheckIfBundleHasIncompatibleKernel) {
  sycl::unittest::UrMock<> Mock;
  // TestKernelWithAspects has GPU aspect, so it shouldn't be compatible with
  // the CPU device and hence shouldn't be in the kernel bundle.
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &redefinedDeviceGetInfoCPU);
  sycl::platform Plt = sycl::platform();
  const sycl::device Dev = Plt.get_devices()[0];
  EXPECT_TRUE(Dev.is_cpu());

  auto Bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      sycl::context(Dev), {Dev});
  auto KernelId1 = sycl::get_kernel_id<TestKernelWithAspects>();
  auto KernelId2 = sycl::get_kernel_id<TestKernel>();

  EXPECT_FALSE(Bundle.has_kernel(KernelId1));
  EXPECT_TRUE(Bundle.has_kernel(KernelId2));
}

TEST(KernelBundle, CheckIfBundleHasCompatibleKernel) {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  // GPU by default.
  const sycl::device Dev = Plt.get_devices()[0];
  EXPECT_TRUE(Dev.is_gpu());

  auto Bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      sycl::context(Dev), {Dev});
  auto KernelId1 = sycl::get_kernel_id<TestKernelWithAspects>();
  auto KernelId2 = sycl::get_kernel_id<TestKernel>();

  EXPECT_TRUE(Bundle.has_kernel(KernelId1));
  EXPECT_TRUE(Bundle.has_kernel(KernelId2));
}

TEST(KernelBundle, CheckIfIncompatibleBundleExists) {
  sycl::unittest::UrMock<> Mock;
  // TestKernelWithAspects has GPU aspect, so it shouldn't be compatible with
  // the CPU device and hence shouldn't be in the kernel bundle.
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &redefinedDeviceGetInfoCPU);
  sycl::platform Plt = sycl::platform();
  const sycl::device Dev = Plt.get_devices()[0];
  EXPECT_TRUE(Dev.is_cpu());

  auto KernelId1 = sycl::get_kernel_id<TestKernelWithAspects>();
  auto KernelId2 = sycl::get_kernel_id<TestKernel>();

  EXPECT_FALSE(sycl::has_kernel_bundle<sycl::bundle_state::executable>(
      sycl::context(Dev), {KernelId1, KernelId2}));
  EXPECT_FALSE(sycl::has_kernel_bundle<sycl::bundle_state::executable>(
      sycl::context(Dev), {KernelId1}));
  EXPECT_TRUE(sycl::has_kernel_bundle<sycl::bundle_state::executable>(
      sycl::context(Dev), {KernelId2}));
}

TEST(KernelBundle, CheckIfCompatibleBundleExists2) {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  // GPU by default.
  const sycl::device Dev = Plt.get_devices()[0];
  EXPECT_TRUE(Dev.is_gpu());

  auto KernelId1 = sycl::get_kernel_id<TestKernelWithAspects>();
  auto KernelId2 = sycl::get_kernel_id<TestKernel>();

  EXPECT_TRUE(sycl::has_kernel_bundle<sycl::bundle_state::executable>(
      sycl::context(Dev), {KernelId1, KernelId2}));
}

TEST(KernelBundle, CheckExceptionIfKernelIncompatible) {
  sycl::unittest::UrMock<> Mock;
  // TestKernelWithAspects has GPU aspect, so it shouldn't be compatible with
  // the CPU device and hence shouldn't be in the kernel bundle.
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &redefinedDeviceGetInfoCPU);
  sycl::platform Plt = sycl::platform();
  const sycl::device Dev = Plt.get_devices()[0];
  EXPECT_TRUE(Dev.is_cpu());

  auto KernelId = sycl::get_kernel_id<TestKernelWithAspects>();
  std::string msg = "";
  try {
    auto Bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
        sycl::context(Dev), {Dev}, {KernelId});
  } catch (sycl::exception &e) {
    msg = e.what();
  }
  EXPECT_EQ(msg, "Kernel is incompatible with all devices in devs");
}

TEST(KernelBundle, HasKernelForSubDevice) {
  sycl::unittest::UrMock<> Mock;

  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &redefinedDeviceGetInfoAfter);
  mock::getCallbacks().set_after_callback("urDevicePartition",
                                          &redefinedDevicePartitionAfter);

  sycl::platform Plt = sycl::platform();
  const sycl::device Dev = Plt.get_devices()[0];

  UrPlatform = sycl::detail::getSyclObjImpl(Plt)->getHandleRef();
  ParentDevice = sycl::detail::getSyclObjImpl(Dev)->getHandleRef();

  sycl::kernel_bundle<sycl::bundle_state::executable> Bundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(
          sycl::context(Dev), {Dev});
  sycl::kernel_id KernelId = sycl::get_kernel_id<TestKernel>();

  EXPECT_TRUE(Bundle.has_kernel(KernelId));

  sycl::device SubDev =
      Dev.create_sub_devices<sycl::info::partition_property::partition_equally>(
          2)[0];

  std::vector<sycl::device> BundleDevs = Bundle.get_devices();
  EXPECT_EQ(std::find(BundleDevs.begin(), BundleDevs.end(), SubDev),
            BundleDevs.end())
      << "Sub-device should not be in the devices of the kernel bundle.";
  EXPECT_FALSE(getSyclObjImpl(SubDev)->isRootDevice());
  EXPECT_TRUE(Bundle.has_kernel(KernelId, SubDev));
}
