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

#include <helpers/MockKernelInfo.hpp>
#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>

#include <gtest/gtest.h>

class TestKernel;
class TestKernelExeOnly;
class TestKernelWithAspects;

MOCK_INTEGRATION_HEADER(TestKernel)
MOCK_INTEGRATION_HEADER(TestKernelExeOnly)
MOCK_INTEGRATION_HEADER(TestKernelWithAspects)

static sycl::unittest::PiImage
generateDefaultImage(std::initializer_list<std::string> KernelNames,
                     pi_device_binary_type BinaryType,
                     const char *DeviceTargetSpec,
                     const std::vector<sycl::aspect> &Aspects = {}) {
  using namespace sycl::unittest;

  PiPropertySet PropSet;
  if (!Aspects.empty())
    addDeviceRequirementsProps(PropSet, Aspects);

  std::vector<unsigned char> Bin{0, 1, 2, 3, 4, 5}; // Random data

  PiArray<PiOffloadEntry> Entries = makeEmptyKernels(KernelNames);

  PiImage Img{BinaryType, // Format
              DeviceTargetSpec,
              "", // Compile options
              "", // Link options
              std::move(Bin),
              std::move(Entries),
              std::move(PropSet)};

  return Img;
}

static sycl::unittest::PiImage Imgs[] = {
    generateDefaultImage({"TestKernel"}, PI_DEVICE_BINARY_TYPE_SPIRV,
                         __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64),
    generateDefaultImage({"TestKernelExeOnly"}, PI_DEVICE_BINARY_TYPE_NATIVE,
                         __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_X86_64),
    // A device image without entires
    generateDefaultImage({}, PI_DEVICE_BINARY_TYPE_NATIVE,
                         __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_X86_64),
    generateDefaultImage(
        {"TestKernelWithAspects"}, PI_DEVICE_BINARY_TYPE_NATIVE,
        __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64, {sycl::aspect::gpu})};
static sycl::unittest::PiImageArray<std::size(Imgs)> ImgArray{Imgs};

static pi_result redefinedDeviceGetInfoCPU(pi_device device,
                                           pi_device_info param_name,
                                           size_t param_value_size,
                                           void *param_value,
                                           size_t *param_value_size_ret) {
  if (param_name == PI_DEVICE_INFO_TYPE) {
    auto *Result = reinterpret_cast<_pi_device_type *>(param_value);
    *Result = PI_DEVICE_TYPE_CPU;
  }
  return PI_SUCCESS;
}

TEST(KernelBundle, GetKernelBundleFromKernel) {
  sycl::unittest::PiMock Mock;

  const sycl::device Dev = Mock.getPlatform().get_devices()[0];
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
  sycl::unittest::PiMock Mock;

  const sycl::device Dev = Mock.getPlatform().get_devices()[0];

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
  sycl::unittest::PiMock Mock;

  const sycl::device Dev = Mock.getPlatform().get_devices()[0];
  sycl::queue Queue{Dev};
  const sycl::context Ctx = Queue.get_context();

  auto EmptyKernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(Ctx, {Dev}, {});

  EXPECT_TRUE(EmptyKernelBundle.empty());
  EXPECT_EQ(std::distance(EmptyKernelBundle.begin(), EmptyKernelBundle.end()),
            0u);
}

TEST(KernelBundle, EmptyKernelBundleKernelLaunchException) {
  sycl::unittest::PiMock Mock;

  const sycl::device Dev = Mock.getPlatform().get_devices()[0];
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
  sycl::unittest::PiMock Mock;

  const sycl::device Dev = Mock.getPlatform().get_devices()[0];

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
  sycl::unittest::PiMock Mock;

  const sycl::device Dev = Mock.getPlatform().get_devices()[0];
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
  sycl::unittest::PiMock Mock;

  const sycl::device Dev = Mock.getPlatform().get_devices()[0];
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
  sycl::unittest::PiMock Mock;

  const sycl::device Dev = Mock.getPlatform().get_devices()[0];
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
  sycl::unittest::PiMock Mock;

  const sycl::device Dev = Mock.getPlatform().get_devices()[0];
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
  sycl::unittest::PiMock Mock;

  const sycl::device Dev = Mock.getPlatform().get_devices()[0];
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

pi_device ParentDevice = nullptr;
pi_platform PiPlatform = nullptr;

pi_result redefinedDeviceGetInfoAfter(pi_device device,
                                      pi_device_info param_name,
                                      size_t param_value_size,
                                      void *param_value,
                                      size_t *param_value_size_ret) {
  if (param_name == PI_DEVICE_INFO_PARTITION_PROPERTIES) {
    if (param_value) {
      auto *Result =
          reinterpret_cast<pi_device_partition_property *>(param_value);
      *Result = PI_DEVICE_PARTITION_EQUALLY;
    }
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(pi_device_partition_property);
  } else if (param_name == PI_DEVICE_INFO_MAX_COMPUTE_UNITS) {
    auto *Result = reinterpret_cast<pi_uint32 *>(param_value);
    *Result = 2;
  } else if (param_name == PI_DEVICE_INFO_PARENT_DEVICE) {
    auto *Result = reinterpret_cast<pi_device *>(param_value);
    *Result = (device == ParentDevice) ? nullptr : ParentDevice;
  } else if (param_name == PI_DEVICE_INFO_PLATFORM) {
    auto *Result = reinterpret_cast<pi_platform *>(param_value);
    *Result = PiPlatform;
  }
  return PI_SUCCESS;
}

pi_result redefinedDevicePartitionAfter(
    pi_device device, const pi_device_partition_property *properties,
    pi_uint32 num_devices, pi_device *out_devices, pi_uint32 *out_num_devices) {
  if (out_devices) {
    for (size_t I = 0; I < num_devices; ++I) {
      out_devices[I] = reinterpret_cast<pi_device>(1000 + I);
    }
  }
  if (out_num_devices)
    *out_num_devices = num_devices;
  return PI_SUCCESS;
}

TEST(KernelBundle, DescendentDevice) {
  // Mock a non-OpenCL plugin since use of descendent devices of context members
  // is not supported there yet.
  sycl::unittest::PiMock Mock(sycl::backend::ext_oneapi_level_zero);

  sycl::platform Plt = Mock.getPlatform();

  PiPlatform = sycl::detail::getSyclObjImpl(Plt)->getHandleRef();

  Mock.redefineAfter<sycl::detail::PiApiKind::piDeviceGetInfo>(
      redefinedDeviceGetInfoAfter);
  Mock.redefineAfter<sycl::detail::PiApiKind::piDevicePartition>(
      redefinedDevicePartitionAfter);

  const sycl::device Dev = Mock.getPlatform().get_devices()[0];
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
  sycl::unittest::PiMock Mock;
  // TestKernelWithAspects has GPU aspect, so it shouldn't be compatible with
  // the CPU device and hence shouldn't be in the kernel bundle.
  Mock.redefineAfter<sycl::detail::PiApiKind::piDeviceGetInfo>(
      redefinedDeviceGetInfoCPU);
  sycl::platform Plt = Mock.getPlatform();
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
  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();
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
  sycl::unittest::PiMock Mock;
  // TestKernelWithAspects has GPU aspect, so it shouldn't be compatible with
  // the CPU device and hence shouldn't be in the kernel bundle.
  Mock.redefineAfter<sycl::detail::PiApiKind::piDeviceGetInfo>(
      redefinedDeviceGetInfoCPU);
  sycl::platform Plt = Mock.getPlatform();
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
  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();
  // GPU by default.
  const sycl::device Dev = Plt.get_devices()[0];
  EXPECT_TRUE(Dev.is_gpu());

  auto KernelId1 = sycl::get_kernel_id<TestKernelWithAspects>();
  auto KernelId2 = sycl::get_kernel_id<TestKernel>();

  EXPECT_TRUE(sycl::has_kernel_bundle<sycl::bundle_state::executable>(
      sycl::context(Dev), {KernelId1, KernelId2}));
}

TEST(KernelBundle, CheckExceptionIfKernelIncompatible) {
  sycl::unittest::PiMock Mock;
  // TestKernelWithAspects has GPU aspect, so it shouldn't be compatible with
  // the CPU device and hence shouldn't be in the kernel bundle.
  Mock.redefineAfter<sycl::detail::PiApiKind::piDeviceGetInfo>(
      redefinedDeviceGetInfoCPU);
  sycl::platform Plt = Mock.getPlatform();
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
  sycl::unittest::PiMock Mock;

  Mock.redefineAfter<sycl::detail::PiApiKind::piDeviceGetInfo>(
      redefinedDeviceGetInfoAfter);
  Mock.redefineAfter<sycl::detail::PiApiKind::piDevicePartition>(
      redefinedDevicePartitionAfter);

  sycl::platform Plt = Mock.getPlatform();
  const sycl::device Dev = Plt.get_devices()[0];

  PiPlatform = sycl::detail::getSyclObjImpl(Plt)->getHandleRef();
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
