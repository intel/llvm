// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"
#include "uur/raii.h"

#include <cstring>

using hipLocalMemOptInTest = uur::urQueueTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE(hipLocalMemOptInTest);

// Module image for the HIP-on-NVIDIA adapter. hipModuleLoadData accepts PTX,
// and the kernel's dynamic shared memory is the launch's local-memory size.
// nvcc -arch=sm_70 emits this shape: the shared symbol is a module-scope
// extern, and its address is a 32-bit shared pointer plus the local offset.
#if defined(__HIP_PLATFORM_NVIDIA__)
static const char *OptInPtx = R"ptx(
.version 8.5
.target sm_70
.address_size 64

.extern .shared .align 16 .b8 smem[];

.visible .entry optin_dyn_smem (
    .param .u64 out,
    .param .u64 local_off,
    .param .u32 nbytes
)
{
    .reg .b16 %rs<5>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd1, [out];
    ld.param.u32 %r1, [nbytes];
    cvta.to.global.u64 %rd2, %rd1;
    ld.param.u32 %r2, [local_off];
    mov.u32 %r3, smem;
    add.s32 %r4, %r3, %r2;
    mov.u16 %rs1, 165;
    st.shared.u8 [%r4], %rs1;
    add.s32 %r5, %r4, %r1;
    mov.u16 %rs2, 90;
    st.shared.u8 [%r5+-1], %rs2;
    ld.shared.u8 %rs3, [%r4];
    st.global.u8 [%rd2], %rs3;
    ld.shared.u8 %rs4, [%r5+-1];
    st.global.u8 [%rd2+1], %rs4;
    ret;
}
)ptx";
#endif

// AMD reports one LDS ceiling, so prepareFunctionLocalMemory returns before
// hipFuncSetAttribute. HIP-on-NVIDIA module functions take the
// cuFuncSetAttribute path once the dynamic size exceeds the default pool.
// This launch is the runtime check. The NVIDIA host wrappers in common.hpp
// are compiled with the adapter and are not executed here.
TEST_P(hipLocalMemOptInTest, LaunchAboveDefaultSharedMemory) {
#ifndef __HIP_PLATFORM_NVIDIA__
  GTEST_SKIP() << "cuFuncSetAttribute is the HIP-on-NVIDIA adapter path";
#else
  ur_native_handle_t native_handle = 0;
  ASSERT_SUCCESS(urDeviceGetNativeHandle(device, &native_handle));
  hipDevice_t hip_device = 0;
  std::memcpy(&hip_device, &native_handle, sizeof(hip_device));

  int default_limit = 0;
  ASSERT_SUCCESS_HIP(hipDeviceGetAttribute(
      &default_limit, hipDeviceAttributeMaxSharedMemoryPerBlock, hip_device));
  ASSERT_GT(default_limit, 0);

  int optin_limit = 0;
  hipError_t optin_status = hipDeviceGetAttribute(
      &optin_limit, hipDeviceAttributeSharedMemPerBlockOptin, hip_device);
  if (optin_status != hipSuccess) {
    (void)hipGetLastError();
    optin_limit = 0;
  }
  if (optin_limit <= 0)
    optin_limit = default_limit;
  if (optin_limit <= default_limit) {
    GTEST_SKIP() << "device has no shared-memory pool above the default limit";
  }

  uint64_t reported = 0;
  ASSERT_SUCCESS(urDeviceGetInfo(device, UR_DEVICE_INFO_LOCAL_MEM_SIZE,
                                 sizeof(reported), &reported, nullptr));
  // The adapter reports the opt-in ceiling. An environment cap may lower it.
  ASSERT_GT(reported, static_cast<uint64_t>(default_limit));
  const size_t bytes = static_cast<size_t>(default_limit) + 1;
  if (bytes > reported) {
    GTEST_SKIP() << "UR_HIP_MAX_LOCAL_MEM_SIZE keeps this launch in the "
                    "default pool";
  }

  uur::raii::Program program = nullptr;
  size_t length = std::strlen(OptInPtx);
  ASSERT_SUCCESS(urProgramCreateWithBinary(context, 1, &device, &length,
                                           (const uint8_t **)&OptInPtx, nullptr,
                                           program.ptr()));
  ASSERT_SUCCESS(urProgramBuild(context, program, nullptr));

  uur::raii::Kernel kernel = nullptr;
  ASSERT_SUCCESS(urKernelCreate(program, "optin_dyn_smem", kernel.ptr()));

  uur::raii::Mem buffer = nullptr;
  ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE, 2, nullptr,
                                   buffer.ptr()));
  unsigned char zeros[2] = {0, 0};
  ASSERT_SUCCESS(urEnqueueMemBufferWrite(queue, buffer, true, 0, 2, zeros, 0,
                                         nullptr, nullptr));

  ur_exp_kernel_arg_value_t out_value = {};
  out_value.memObjTuple = {buffer, UR_MEM_FLAG_READ_WRITE};
  ur_exp_kernel_arg_properties_t out_arg = {
      UR_STRUCTURE_TYPE_EXP_KERNEL_ARG_PROPERTIES,
      nullptr,
      UR_EXP_KERNEL_ARG_TYPE_MEM_OBJ,
      0,
      sizeof(ur_mem_handle_t),
      out_value};

  ur_exp_kernel_arg_properties_t local_arg = {
      UR_STRUCTURE_TYPE_EXP_KERNEL_ARG_PROPERTIES,
      nullptr,
      UR_EXP_KERNEL_ARG_TYPE_LOCAL,
      1,
      bytes,
      {nullptr}};

  const uint32_t nbytes = static_cast<uint32_t>(bytes);
  ur_exp_kernel_arg_value_t size_value = {};
  size_value.value = &nbytes;
  ur_exp_kernel_arg_properties_t size_arg = {
      UR_STRUCTURE_TYPE_EXP_KERNEL_ARG_PROPERTIES,
      nullptr,
      UR_EXP_KERNEL_ARG_TYPE_VALUE,
      2,
      sizeof(nbytes),
      size_value};

  ur_exp_kernel_arg_properties_t args[] = {out_arg, local_arg, size_arg};
  const size_t offset[] = {0};
  const size_t global[] = {1};
  const size_t local[] = {1};
  ur_result_t launch =
      urEnqueueKernelLaunchWithArgsExp(queue, kernel, 1, offset, global, local,
                                       3, args, nullptr, 0, nullptr, nullptr);
  if (launch != UR_RESULT_SUCCESS) {
    const char *message = nullptr;
    int32_t code = 0;
    (void)urAdapterGetLastError(adapter, &message, &code);
    FAIL() << "opt-in launch failed: " << launch << " adapter code " << code
           << " " << (message ? message : "");
  }
  ASSERT_SUCCESS(urQueueFinish(queue));

  unsigned char got[2] = {0, 0};
  ASSERT_SUCCESS(urEnqueueMemBufferRead(queue, buffer, true, 0, 2, got, 0,
                                        nullptr, nullptr));
  EXPECT_EQ(got[0], 0xA5);
  EXPECT_EQ(got[1], 0x5A);
#endif
}
