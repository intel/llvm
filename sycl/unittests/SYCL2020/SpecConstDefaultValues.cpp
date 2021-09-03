//==---- DefaultValues.cpp --- Spec constants default values unit test -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define SYCL2020_DISABLE_DEPRECATION_WARNINGS

#include <CL/sycl.hpp>
#include <detail/device_image_impl.hpp>

#include <helpers/CommonRedefinitions.hpp>
#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>

#include <gtest/gtest.h>

class TestKernel;
const static sycl::specialization_id<int> SpecConst1{42};

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
template <> struct KernelInfo<TestKernel> {
  static constexpr unsigned getNumParams() { return 0; }
  static const kernel_param_desc_t &getParamDesc(int) {
    static kernel_param_desc_t Dummy;
    return Dummy;
  }
  static constexpr const char *getName() {
    return "SpecConstDefaultValues_TestKernel";
  }
  static constexpr bool isESIMD() { return false; }
  static constexpr bool callsThisItem() { return false; }
  static constexpr bool callsAnyThisFreeFunction() { return false; }
};

template <> const char *get_spec_constant_symbolic_ID<SpecConst1>() {
  return "SC1";
}
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

static sycl::unittest::PiImage generateImageWithSpecConsts() {
  using namespace sycl::unittest;

  std::vector<char> SpecConstData;
  PiProperty SC1 = makeSpecConstant<int>(SpecConstData, "SC1", {0}, {0}, {42});
  PiProperty SC2 = makeSpecConstant<int>(SpecConstData, "SC2", {1}, {0}, {8});

  PiPropertySet PropSet;
  addSpecConstants({SC1, SC2}, std::move(SpecConstData), PropSet);

  std::vector<unsigned char> Bin{0, 1, 2, 3, 4, 5}; // Random data

  PiArray<PiOffloadEntry> Entries =
      makeEmptyKernels({"SpecConstDefaultValues_TestKernel"});

  PiImage Img{PI_DEVICE_BINARY_TYPE_SPIRV,            // Format
              __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64, // DeviceTargetSpec
              "",                                     // Compile options
              "",                                     // Link options
              std::move(Bin),
              std::move(Entries),
              std::move(PropSet)};

  return Img;
}

static sycl::unittest::PiImage Img = generateImageWithSpecConsts();
static sycl::unittest::PiImageArray<1> ImgArray{&Img};

TEST(SpecConstDefaultValues, DefaultValuesAreSet) {
  sycl::platform Plt{sycl::default_selector()};
  if (Plt.is_host()) {
    std::cerr << "Test is not supported on host, skipping\n";
    return; // test is not supported on host.
  }

  if (Plt.get_backend() == sycl::backend::cuda) {
    std::cerr << "Test is not supported on CUDA platform, skipping\n";
    return;
  }

  if (Plt.get_backend() == sycl::backend::rocm) {
    std::cerr << "Test is not supported on ROCm platform, skipping\n";
    return;
  }

  sycl::unittest::PiMock Mock{Plt};
  setupDefaultMockAPIs(Mock);

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

TEST(SpecConstDefaultValues, DefaultValuesAreOverriden) {
  sycl::platform Plt{sycl::default_selector()};
  if (Plt.is_host()) {
    std::cerr << "Test is not supported on host, skipping\n";
    return; // test is not supported on host.
  }

  if (Plt.get_backend() == sycl::backend::cuda) {
    std::cerr << "Test is not supported on CUDA platform, skipping\n";
    return;
  }

  if (Plt.get_backend() == sycl::backend::rocm) {
    std::cerr << "Test is not supported on ROCm platform, skipping\n";
    return;
  }

  sycl::unittest::PiMock Mock{Plt};
  setupDefaultMockAPIs(Mock);

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
