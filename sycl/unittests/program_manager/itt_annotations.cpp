//==---- itt_annotations.cpp --- ITT Annotations for SPIR-V images ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define SYCL2020_DISABLE_DEPRECATION_WARNINGS

#include <CL/sycl.hpp>
#include <detail/config.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>

#include <gtest/gtest.h>

#include <stdlib.h>

// Same as defined in config.def
static constexpr auto ITTProfileEnvVarName = "INTEL_ENABLE_OFFLOAD_ANNOTATIONS";

static void set_env(const char *name, const char *value) {
#ifdef _WIN32
  (void)_putenv_s(name, value);
#else
  (void)setenv(name, value, /*overwrite*/ 1);
#endif
}

static void unset_env(const char *name) {
#ifdef _WIN32
  (void)_putenv_s(name, "");
#else
  unsetenv(name);
#endif
}

class TestKernel;

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

static pi_result redefinedKernelGetGroupInfo(pi_kernel kernel, pi_device device,
                                             pi_kernel_group_info param_name,
                                             size_t param_value_size,
                                             void *param_value,
                                             size_t *param_value_size_ret) {
  if (param_name == PI_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE) {
    if (param_value_size_ret) {
      *param_value_size_ret = 3 * sizeof(size_t);
    } else if (param_value) {
      auto size = static_cast<size_t *>(param_value);
      size[0] = 0;
      size[1] = 0;
      size[2] = 0;
    }
  }

  return PI_SUCCESS;
}

bool HasITTEnabled = false;

static pi_result
redefinedProgramSetSpecializationConstant(pi_program prog, pi_uint32 spec_id,
                                          size_t spec_size,
                                          const void *spec_value) {
  if (spec_id == sycl::detail::ITTSpecConstId)
    HasITTEnabled = true;

  return PI_SUCCESS;
}

static pi_result redefinedEnqueueKernelLaunch(pi_queue, pi_kernel, pi_uint32,
                                              const size_t *, const size_t *,
                                              const size_t *, pi_uint32,
                                              const pi_event *, pi_event *) {
  return PI_SUCCESS;
}

static void reset() {
  using namespace sycl::detail;
  HasITTEnabled = false;
  SYCLConfig<INTEL_ENABLE_OFFLOAD_ANNOTATIONS>::reset();
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
  Mock.redefine<PiApiKind::piKernelGetGroupInfo>(redefinedKernelGetGroupInfo);
}

static sycl::unittest::PiImage generateDefaultImage() {
  using namespace sycl::unittest;

  PiPropertySet PropSet;

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
sycl::unittest::PiImageArray<1> ImgArray{&Img};

TEST(ITTNotify, UseKernelBundle) {
  set_env(ITTProfileEnvVarName, "1");

  reset();

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
  auto ExecBundle = sycl::build(KernelBundle);
  Queue.submit([&](sycl::handler &CGH) {
    CGH.use_kernel_bundle(ExecBundle);
    CGH.single_task<TestKernel>([] {}); // Actual kernel does not matter
  });

  EXPECT_EQ(HasITTEnabled, true);
}

TEST(ITTNotify, VarNotSet) {
  unset_env(ITTProfileEnvVarName);

  reset();

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
  auto ExecBundle = sycl::build(KernelBundle);
  Queue.submit([&](sycl::handler &CGH) {
    CGH.use_kernel_bundle(ExecBundle);
    CGH.single_task<TestKernel>([] {}); // Actual kernel does not matter
  });

  EXPECT_EQ(HasITTEnabled, false);
}
