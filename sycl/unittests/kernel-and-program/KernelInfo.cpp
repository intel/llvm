//==-------------- KernelInfo.cpp --- kernel info unit test ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <gtest/gtest.h>
#include <helpers/PiMock.hpp>

using namespace sycl;

static pi_result
redefinedProgramBuild(pi_program program, pi_uint32 num_devices,
                      const pi_device *device_list, const char *options,
                      void (*pfn_notify)(pi_program program, void *user_data),
                      void *user_data) {
  return PI_SUCCESS;
}

static pi_result redefinedProgramCompile(
    pi_program program, pi_uint32 num_devices, const pi_device *device_list,
    const char *options, pi_uint32 num_input_headers,
    const pi_program *input_headers, const char **header_include_names,
    void (*pfn_notify)(pi_program program, void *user_data), void *user_data) {
  return PI_SUCCESS;
}

static pi_result
redefinedProgramLink(pi_context context, pi_uint32 num_devices,
                     const pi_device *device_list, const char *options,
                     pi_uint32 num_input_programs,
                     const pi_program *input_programs,
                     void (*pfn_notify)(pi_program program, void *user_data),
                     void *user_data, pi_program *ret_program) {
  return PI_SUCCESS;
}

static pi_result redefinedKernelCreate(pi_program program,
                                       const char *kernel_name,
                                       pi_kernel *ret_kernel) {
  return PI_SUCCESS;
}

static pi_result redefinedKernelRetain(pi_kernel kernel) { return PI_SUCCESS; }

static pi_result redefinedKernelRelease(pi_kernel kernel) { return PI_SUCCESS; }

static pi_result redefinedKernelGetInfo(pi_kernel kernel,
                                        pi_kernel_info param_name,
                                        size_t param_value_size,
                                        void *param_value,
                                        size_t *param_value_size_ret) {
  return PI_SUCCESS;
}

bool privateMemSizeCalled = false;

static pi_result redefinedKernelGetGroupInfo(pi_kernel kernel, pi_device device,
                               pi_kernel_group_info param_name,
                               size_t param_value_size, void *param_value,
                               size_t *param_value_size_ret) {
  if (param_name == PI_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE) {
    privateMemSizeCalled = true;
  }

  return PI_SUCCESS;
}

static pi_result redefinedKernelSetExecInfo(pi_kernel kernel,
                                            pi_kernel_exec_info value_name,
                                            size_t param_value_size,
                                            const void *param_value) {
  return PI_SUCCESS;
}

class KernelInfoTest : public ::testing::Test {
public:
  KernelInfoTest() : Plt{default_selector()} {}

protected:
  void SetUp() override {
    if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
      std::clog << "This test is only supported on OpenCL devices\n";
      std::clog << "Current platform is "
                << Plt.get_info<info::platform::name>();
      return;
    }

    Mock = std::make_unique<unittest::PiMock>(Plt);

    Mock->redefine<detail::PiApiKind::piProgramCompile>(
        redefinedProgramCompile);
    Mock->redefine<detail::PiApiKind::piProgramLink>(redefinedProgramLink);
    Mock->redefine<detail::PiApiKind::piProgramBuild>(redefinedProgramBuild);
    Mock->redefine<detail::PiApiKind::piKernelCreate>(redefinedKernelCreate);
    Mock->redefine<detail::PiApiKind::piKernelRetain>(redefinedKernelRetain);
    Mock->redefine<detail::PiApiKind::piKernelRelease>(redefinedKernelRelease);
    Mock->redefine<detail::PiApiKind::piKernelGetInfo>(redefinedKernelGetInfo);
    Mock->redefine<detail::PiApiKind::piKernelSetExecInfo>(
        redefinedKernelSetExecInfo);
    Mock->redefine<detail::PiApiKind::piKernelGetGroupInfo>(
        redefinedKernelGetGroupInfo);
  }

protected:
  platform Plt;
  std::unique_ptr<unittest::PiMock> Mock;
};

class TestKernel {
public:
  void operator()(cl::sycl::item<1>){};
};

TEST_F(KernelInfoTest, GetPrivateMemUsage) {
  if (Plt.is_host()) {
    return;
  }

  context Ctx{Plt};
  program Prg{Ctx};

  Prg.build_with_kernel_type<TestKernel>();
  kernel Ker = Prg.get_kernel<TestKernel>();
  Ker.get_info<info::kernel_device_specific::private_mem_size>(
      Ctx.get_devices()[0]);
  EXPECT_EQ(privateMemSizeCalled, true) << "Expect piKernelGetInfo to be "
    << "called with PI_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE";
}
