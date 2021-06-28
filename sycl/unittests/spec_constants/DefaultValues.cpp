//==---- DefaultValues.cpp --- Spec constants default values unit test -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define SYCL2020_DISABLE_DEPRECATION_WARNINGS

#include <CL/sycl.hpp>
#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>

#include <gtest/gtest.h>

class TestKernel;
class TestKernel2;
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

static pi_result redefinedProgramCreate(pi_context, const void *, size_t,
                                        pi_program *) {
  return PI_SUCCESS;
}

static pi_result redefinedProgramBuild(
    pi_program prog, pi_uint32, const pi_device *, const char *,
    void (*pfn_notify)(pi_program program, void *user_data), void *user_data) {
  if (pfn_notify) {
    pfn_notify(prog, user_data);
  }
  return PI_SUCCESS;
}

static pi_result redefinedProgramCompile(pi_program, pi_uint32,
                                         const pi_device *, const char *,
                                         pi_uint32, const pi_program *,
                                         const char **,
                                         void (*)(pi_program, void *), void *) {
  return PI_SUCCESS;
}

static pi_result redefinedProgramLink(pi_context, pi_uint32, const pi_device *,
                                      const char *, pi_uint32,
                                      const pi_program *,
                                      void (*)(pi_program, void *), void *,
                                      pi_program *) {
  return PI_SUCCESS;
}

static pi_result redefinedProgramGetInfo(pi_program program,
                                         pi_program_info param_name,
                                         size_t param_value_size,
                                         void *param_value,
                                         size_t *param_value_size_ret) {
  if (param_name == PI_PROGRAM_INFO_NUM_DEVICES) {
    auto value = reinterpret_cast<unsigned int *>(param_value);
    *value = 1;
  }

  if (param_name == PI_PROGRAM_INFO_BINARY_SIZES) {
    auto value = reinterpret_cast<size_t *>(param_value);
    value[0] = 1;
  }

  if (param_name == PI_PROGRAM_INFO_BINARIES) {
    auto value = reinterpret_cast<unsigned char *>(param_value);
    value[0] = 1;
  }

  return PI_SUCCESS;
}

static pi_result redefinedProgramRetain(pi_program program) {
  return PI_SUCCESS;
}

static pi_result redefinedProgramRelease(pi_program program) {
  return PI_SUCCESS;
}

static pi_result redefinedKernelCreate(pi_program program,
                                       const char *kernel_name,
                                       pi_kernel *ret_kernel) {
  *ret_kernel = reinterpret_cast<pi_kernel>(new int[1]);
  return PI_SUCCESS;
}

static pi_result redefinedKernelRetain(pi_kernel kernel) { return PI_SUCCESS; }

static pi_result redefinedKernelRelease(pi_kernel kernel) {
  delete[] reinterpret_cast<int *>(kernel);
  return PI_SUCCESS;
}

static pi_result redefinedKernelGetInfo(pi_kernel kernel,
                                        pi_kernel_info param_name,
                                        size_t param_value_size,
                                        void *param_value,
                                        size_t *param_value_size_ret) {
  return PI_SUCCESS;
}

static pi_result redefinedKernelSetExecInfo(pi_kernel kernel,
                                            pi_kernel_exec_info value_name,
                                            size_t param_value_size,
                                            const void *param_value) {
  return PI_SUCCESS;
}

static pi_result redefinedEventsWait(pi_uint32 num_events,
                                     const pi_event *event_list) {
  return PI_SUCCESS;
}

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

static pi_result redefinedEnqueueKernelLaunch(pi_queue, pi_kernel, pi_uint32,
                                              const size_t *, const size_t *,
                                              const size_t *, pi_uint32,
                                              const pi_event *, pi_event *) {
  return PI_SUCCESS;
}

static void setupDefaultMockAPIs(sycl::unittest::PiMock &Mock) {
  using namespace sycl::detail;
  Mock.redefine<PiApiKind::piProgramCreate>(redefinedProgramCreate);
  Mock.redefine<PiApiKind::piProgramCompile>(redefinedProgramCompile);
  Mock.redefine<PiApiKind::piProgramLink>(redefinedProgramLink);
  Mock.redefine<PiApiKind::piProgramBuild>(redefinedProgramBuild);
  Mock.redefine<PiApiKind::piProgramGetInfo>(redefinedProgramGetInfo);
  Mock.redefine<PiApiKind::piProgramRetain>(redefinedProgramRetain);
  Mock.redefine<PiApiKind::piProgramRelease>(redefinedProgramRelease);
  Mock.redefine<PiApiKind::piKernelCreate>(redefinedKernelCreate);
  Mock.redefine<PiApiKind::piKernelRetain>(redefinedKernelRetain);
  Mock.redefine<PiApiKind::piKernelRelease>(redefinedKernelRelease);
  Mock.redefine<PiApiKind::piKernelGetInfo>(redefinedKernelGetInfo);
  Mock.redefine<PiApiKind::piKernelSetExecInfo>(redefinedKernelSetExecInfo);
  Mock.redefine<PiApiKind::piextProgramSetSpecializationConstant>(
      redefinedProgramSetSpecializationConstant);
  Mock.redefine<PiApiKind::piEventsWait>(redefinedEventsWait);
  Mock.redefine<PiApiKind::piEnqueueKernelLaunch>(redefinedEnqueueKernelLaunch);
}

static sycl::unittest::PiImage generateDefaultImage() {
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

sycl::unittest::PiImage Img = generateDefaultImage();
sycl::unittest::PiImageArray ImgArray{Img};

TEST(DefaultValues, DISABLED_DefaultValuesAreSet) {
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

TEST(DefaultValues, DISABLED_DefaultValuesAreOverriden) {
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
