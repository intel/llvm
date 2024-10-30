//==------ SpecializationConstant.cpp --- Spec constants unit tests --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define SYCL2020_DISABLE_DEPRECATION_WARNINGS

#include <detail/device_image_impl.hpp>
#include <sycl/sycl.hpp>

#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

class TestKernel;
const static sycl::specialization_id<int> SpecConst1{42};

namespace sycl {
inline namespace _V1 {
namespace detail {
template <>
struct KernelInfo<TestKernel> : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() {
    return "SpecializationConstant_TestKernel";
  }
};

template <> const char *get_spec_constant_symbolic_ID<SpecConst1>() {
  return "SC1";
}
} // namespace detail
} // namespace _V1
} // namespace sycl

static sycl::unittest::MockDeviceImage generateImageWithSpecConsts() {
  using namespace sycl::unittest;

  std::vector<char> SpecConstData;
  MockProperty SC1 =
      makeSpecConstant<int>(SpecConstData, "SC1", {0}, {0}, {42});
  MockProperty SC2 = makeSpecConstant<int>(SpecConstData, "SC2", {1}, {0}, {8});

  MockPropertySet PropSet;
  addSpecConstants({SC1, SC2}, std::move(SpecConstData), PropSet);

  std::vector<MockOffloadEntry> Entries =
      makeEmptyKernels({"SpecializationConstant_TestKernel"});
  MockDeviceImage Img(std::move(Entries), std::move(PropSet));

  return Img;
}

static sycl::unittest::MockDeviceImage Img = generateImageWithSpecConsts();
static sycl::unittest::MockDeviceImageArray<1> ImgArray{&Img};

TEST(SpecializationConstant, DefaultValuesAreSet) {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();

  const sycl::device Dev = Plt.get_devices()[0];

  sycl::queue Queue{Dev};

  const sycl::context Ctx = Queue.get_context();

  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev});

  sycl::kernel_id TestKernelID = sycl::get_kernel_id<TestKernel>();
  auto DevImage =
      std::find_if(KernelBundle.begin(), KernelBundle.end(),
                   [&](auto Image) { return Image.has_kernel(TestKernelID); });
  EXPECT_NE(DevImage, KernelBundle.end());

  auto DevImageImpl = sycl::detail::getSyclObjImpl(*DevImage);
  const auto &Blob = DevImageImpl->get_spec_const_blob_ref();

  int SpecConstVal1 = *reinterpret_cast<const int *>(Blob.data());
  int SpecConstVal2 = *(reinterpret_cast<const int *>(Blob.data()) + 1);

  EXPECT_EQ(SpecConstVal1, 42);
  EXPECT_EQ(SpecConstVal2, 8);
}

TEST(SpecializationConstant, DefaultValuesAreOverriden) {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();

  const sycl::device Dev = Plt.get_devices()[0];

  sycl::queue Queue{Dev};

  const sycl::context Ctx = Queue.get_context();

  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev});

  sycl::kernel_id TestKernelID = sycl::get_kernel_id<TestKernel>();
  auto DevImage =
      std::find_if(KernelBundle.begin(), KernelBundle.end(),
                   [&](auto Image) { return Image.has_kernel(TestKernelID); });
  EXPECT_NE(DevImage, KernelBundle.end());

  auto DevImageImpl = sycl::detail::getSyclObjImpl(*DevImage);
  auto &Blob = DevImageImpl->get_spec_const_blob_ref();
  int SpecConstVal1 = *reinterpret_cast<int *>(Blob.data());
  int SpecConstVal2 = *(reinterpret_cast<int *>(Blob.data()) + 1);

  EXPECT_EQ(SpecConstVal1, 42);
  EXPECT_EQ(SpecConstVal2, 8);

  KernelBundle.set_specialization_constant<SpecConst1>(80);

  SpecConstVal1 = *reinterpret_cast<int *>(Blob.data());
  SpecConstVal2 = *(reinterpret_cast<int *>(Blob.data()) + 1);

  EXPECT_EQ(SpecConstVal1, 80);
  EXPECT_EQ(SpecConstVal2, 8);
}

TEST(SpecializationConstant, SetSpecConstAfterUseKernelBundle) {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();

  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};
  sycl::queue Queue{Ctx, Dev};

  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(Ctx, {Dev});

  // Create uniquely identifyable class to throw on expected exception
  class UniqueException {};

  try {
    Queue.submit([&](sycl::handler &CGH) {
      CGH.use_kernel_bundle(KernelBundle);
      try {
        CGH.set_specialization_constant<SpecConst1>(80);
        FAIL() << "No exception was thrown.";
      } catch (const sycl::exception &e) {
        if (static_cast<sycl::errc>(e.code().value()) != sycl::errc::invalid) {
          FAIL() << "Unexpected SYCL exception was thrown.";
          throw;
        }
        throw UniqueException{};
      } catch (...) {
        FAIL() << "Unexpected non-SYCL exception was thrown.";
        throw;
      }
      CGH.single_task<TestKernel>([]() {});
    });
  } catch (const sycl::exception &e) {
    if (static_cast<sycl::errc>(e.code().value()) == sycl::errc::invalid) {
      FAIL() << "SYCL exception with error code sycl::errc::invalid was "
                "thrown at the wrong level.";
    }
    throw;
  } catch (const UniqueException &) {
    // Expected path
  }
}

TEST(SpecializationConstant, GetSpecConstAfterUseKernelBundle) {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();

  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};
  sycl::queue Queue{Ctx, Dev};

  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(Ctx, {Dev});

  // Create uniquely identifyable class to throw on expected exception
  class UniqueException {};

  try {
    Queue.submit([&](sycl::handler &CGH) {
      CGH.use_kernel_bundle(KernelBundle);
      try {
        auto SpecConst1Val = CGH.get_specialization_constant<SpecConst1>();
        (void)SpecConst1Val;
        FAIL() << "No exception was thrown.";
      } catch (const sycl::exception &e) {
        if (static_cast<sycl::errc>(e.code().value()) != sycl::errc::invalid) {
          FAIL() << "Unexpected SYCL exception was thrown.";
          throw;
        }
        throw UniqueException{};
      } catch (...) {
        FAIL() << "Unexpected non-SYCL exception was thrown.";
        throw;
      }
      CGH.single_task<TestKernel>([]() {});
    });
  } catch (const sycl::exception &e) {
    if (static_cast<sycl::errc>(e.code().value()) == sycl::errc::invalid) {
      FAIL() << "SYCL exception with error code sycl::errc::invalid was "
                "thrown at the wrong level.";
    }
    throw;
  } catch (const UniqueException &) {
    // Expected path
  }
}

TEST(SpecializationConstant, UseKernelBundleAfterSetSpecConst) {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();

  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};
  sycl::queue Queue{Ctx, Dev};

  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(Ctx, {Dev});

  // Create uniquely identifyable class to throw on expected exception
  class UniqueException {};

  try {
    Queue.submit([&](sycl::handler &CGH) {
      CGH.set_specialization_constant<SpecConst1>(80);
      try {
        CGH.use_kernel_bundle(KernelBundle);
        FAIL() << "No exception was thrown.";
      } catch (const sycl::exception &e) {
        if (static_cast<sycl::errc>(e.code().value()) != sycl::errc::invalid) {
          FAIL() << "Unexpected SYCL exception was thrown.";
          throw;
        }
        throw UniqueException{};
      } catch (...) {
        FAIL() << "Unexpected non-SYCL exception was thrown.";
        throw;
      }
      CGH.single_task<TestKernel>([]() {});
    });
  } catch (const sycl::exception &e) {
    if (static_cast<sycl::errc>(e.code().value()) == sycl::errc::invalid) {
      FAIL() << "SYCL exception with error code sycl::errc::invalid was "
                "thrown at the wrong level.";
    }
    throw;
  } catch (const UniqueException &) {
    // Expected path
  }
}

TEST(SpecializationConstant, NoKernel) {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();

  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};
  sycl::queue Queue{Ctx, Dev};

  Queue.submit([&](sycl::handler &CGH) {
    int ExpectedValue = 42;
    CGH.set_specialization_constant<SpecConst1>(ExpectedValue);
    EXPECT_EQ(CGH.get_specialization_constant<SpecConst1>(), ExpectedValue);
  });
}
