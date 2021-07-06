//==---- DefaultValues.cpp --- Spec constants default values unit test -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define SYCL2020_DISABLE_DEPRECATION_WARNINGS

#include <CL/sycl.hpp>

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
  static constexpr const char *getName() { return "TestKernel"; }
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

int SpecConstVal0 = 0;
int SpecConstVal1 = 0;

static pi_result
redefinedProgramSetSpecializationConstant(pi_program prog, pi_uint32 spec_id,
                                          size_t spec_size,
                                          const void *spec_value) {
  if (spec_id == 0)
    SpecConstVal0 = *static_cast<const int *>(spec_value);
  if (spec_id == 1)
    SpecConstVal1 = *static_cast<const int *>(spec_value);

  return PI_SUCCESS;
}

static sycl::unittest::PiImage generateImageWithSpecConsts() {
  using namespace sycl::unittest;

  std::vector<char> SpecConstData;
  PiProperty SC1 = makeSpecConstant<int>(SpecConstData, "SC1", {0}, {0}, {42});
  PiProperty SC2 = makeSpecConstant<int>(SpecConstData, "SC2", {1}, {0}, {8});

  PiPropertySet PropSet;
  addSpecConstants({SC1, SC2}, std::move(SpecConstData), PropSet);

  std::vector<unsigned char> Bin{0, 1, 2, 3, 4, 5}; // Random data

  PiArray<PiOffloadEntry> Entries = makeEmptyKernels({"TestKernel"});

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

TEST(SpecConstDefaultValues, DISABLED_DefaultValuesAreSet) {
  sycl::platform Plt{sycl::default_selector()};
  if (Plt.is_host()) {
    std::cerr << "Test is not supported on host, skipping\n";
    return; // test is not supported on host.
  }

  if (Plt.get_backend() == sycl::backend::cuda) {
    std::cerr << "Test is not supported on CUDA platform, skipping\n";
    return;
  }

  sycl::unittest::PiMock Mock{Plt};
  setupDefaultMockAPIs(Mock);
  Mock.redefine<sycl::detail::PiApiKind::piextProgramSetSpecializationConstant>(
      redefinedProgramSetSpecializationConstant);

  const sycl::device Dev = Plt.get_devices()[0];

  sycl::queue Queue{Dev};

  const sycl::context Ctx = Queue.get_context();

  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev});
  auto ExecBundle = sycl::build(KernelBundle);
  Queue.submit([&](sycl::handler &CGH) {
    CGH.use_kernel_bundle(ExecBundle);
    CGH.single_task<TestKernel>([] {}); // Actual kernel does not matter
  });

  EXPECT_EQ(SpecConstVal0, 42);
  EXPECT_EQ(SpecConstVal1, 8);
}

TEST(SpecConstDefaultValues, DISABLED_DefaultValuesAreOverriden) {
  sycl::platform Plt{sycl::default_selector()};
  if (Plt.is_host()) {
    std::cerr << "Test is not supported on host, skipping\n";
    return; // test is not supported on host.
  }

  if (Plt.get_backend() == sycl::backend::cuda) {
    std::cerr << "Test is not supported on CUDA platform, skipping\n";
    return;
  }

  sycl::unittest::PiMock Mock{Plt};
  setupDefaultMockAPIs(Mock);
  Mock.redefine<sycl::detail::PiApiKind::piextProgramSetSpecializationConstant>(
      redefinedProgramSetSpecializationConstant);

  const sycl::device Dev = Plt.get_devices()[0];

  sycl::queue Queue{Dev};

  const sycl::context Ctx = Queue.get_context();

  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev});
  KernelBundle.set_specialization_constant<SpecConst1>(80);
  auto ExecBundle = sycl::build(KernelBundle);
  Queue.submit([&](sycl::handler &CGH) {
    CGH.use_kernel_bundle(ExecBundle);
    CGH.single_task<TestKernel>([] {}); // Actual kernel does not matter
  });

  EXPECT_EQ(SpecConstVal0, 80);
  EXPECT_EQ(SpecConstVal1, 8);
}
