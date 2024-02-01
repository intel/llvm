// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "kernel.hpp"
#include "uur/fixtures.h"
#include "uur/raii.h"

using cudaKernelTest = uur::urQueueTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(cudaKernelTest);

// The first argument stores the implicit global offset
inline constexpr size_t NumberOfImplicitArgsCUDA = 1;

const char *ptxSource = "\n\
.version 3.2\n\
.target sm_20\n\
.address_size 64\n\
.visible .entry _Z8myKernelPi(\n\
	.param .u64 _Z8myKernelPi_param_0\n\
)\n\
{\n\
	.reg .s32 	%r<5>;\n\
	.reg .s64 	%rd<5>;\n\
	ld.param.u64 	%rd1, [_Z8myKernelPi_param_0];\n\
	cvta.to.global.u64 	%rd2, %rd1;\n\
	.loc 1 3 1\n\
	mov.u32 	%r1, %ntid.x;\n\
	mov.u32 	%r2, %ctaid.x;\n\
	mov.u32 	%r3, %tid.x;\n\
	mad.lo.s32 	%r4, %r1, %r2, %r3;\n\
	mul.wide.s32 	%rd3, %r4, 4;\n\
	add.s64 	%rd4, %rd2, %rd3;\n\
	.loc 1 4 1\n\
	st.global.u32 	[%rd4], %r4;\n\
	.loc 1 5 2\n\
	ret;\n\
    ret;\
\n\
}\
\n\
";

const char *twoParams = "\n\
.version 3.2\n\
.target sm_20\n\
.address_size 64\n\
.visible .entry twoParamKernel(\n\
	.param .u64 twoParamKernel_param_0,\n\
  .param .u64 twoParamKernel_param_1\n\
)\n\
{\n\
  ret;\
  \n\
}\n\
";

const char *threeParamsTwoLocal = "\n\
.version 3.2\n\
.target sm_20\n\
.address_size 64\n\
.visible .entry twoParamKernelLocal(\n\
	.param .u64 twoParamKernel_param_0,\n\
  .param .u32 twoParamKernel_param_1,\n\
  .param .u32 twoParamKernel_param_2\n\
)\n\
{\n\
  ret;\
  \n\
}\n\
";

TEST_P(cudaKernelTest, CreateProgramAndKernel) {

    uur::raii::Program program = nullptr;
    ASSERT_SUCCESS(urProgramCreateWithBinary(
        context, device, std::strlen(ptxSource), (const uint8_t *)ptxSource,
        nullptr, program.ptr()));
    ASSERT_NE(program, nullptr);
    ASSERT_SUCCESS(urProgramBuild(context, program, nullptr));

    uur::raii::Kernel kernel = nullptr;
    ASSERT_SUCCESS(urKernelCreate(program, "_Z8myKernelPi", kernel.ptr()));
    ASSERT_NE(kernel, nullptr);
}

TEST_P(cudaKernelTest, CreateProgramAndKernelWithMetadata) {

    std::vector<uint32_t> reqdWorkGroupSizeMD;
    reqdWorkGroupSizeMD.reserve(5);
    // 64-bit representing bit size
    reqdWorkGroupSizeMD.push_back(96);
    reqdWorkGroupSizeMD.push_back(0);
    // reqd_work_group_size x
    reqdWorkGroupSizeMD.push_back(8);
    // reqd_work_group_size y
    reqdWorkGroupSizeMD.push_back(16);
    // reqd_work_group_size z
    reqdWorkGroupSizeMD.push_back(32);

    const char *reqdWorkGroupSizeMDConstName =
        "_Z8myKernelPi@reqd_work_group_size";
    std::vector<char> reqdWorkGroupSizeMDName(
        reqdWorkGroupSizeMDConstName, reqdWorkGroupSizeMDConstName +
                                          strlen(reqdWorkGroupSizeMDConstName) +
                                          1);

    ur_program_metadata_value_t reqd_work_group_value;
    reqd_work_group_value.pData = reqdWorkGroupSizeMD.data();

    ur_program_metadata_t reqdWorkGroupSizeMDProp = {
        reqdWorkGroupSizeMDName.data(), UR_PROGRAM_METADATA_TYPE_BYTE_ARRAY,
        reqdWorkGroupSizeMD.size() * sizeof(uint32_t), reqd_work_group_value};

    ur_program_properties_t programProps{UR_STRUCTURE_TYPE_PROGRAM_PROPERTIES,
                                         nullptr, 1, &reqdWorkGroupSizeMDProp};
    uur::raii::Program program = nullptr;
    ASSERT_SUCCESS(urProgramCreateWithBinary(
        context, device, std::strlen(ptxSource), (const uint8_t *)ptxSource,
        &programProps, program.ptr()));
    ASSERT_NE(program, nullptr);

    ASSERT_SUCCESS(urProgramBuild(context, program, nullptr));

    uur::raii::Kernel kernel = nullptr;
    ASSERT_SUCCESS(urKernelCreate(program, "_Z8myKernelPi", kernel.ptr()));

    size_t compileWGSize[3] = {0};
    ASSERT_SUCCESS(urKernelGetGroupInfo(
        kernel, device, UR_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE,
        sizeof(compileWGSize), &compileWGSize, nullptr));

    for (int i = 0; i < 3; i++) {
        ASSERT_EQ(compileWGSize[i], reqdWorkGroupSizeMD[i + 2]);
    }
}

TEST_P(cudaKernelTest, URKernelArgumentSimple) {
    uur::raii::Program program = nullptr;
    ASSERT_SUCCESS(urProgramCreateWithBinary(
        context, device, std::strlen(ptxSource), (const uint8_t *)ptxSource,
        nullptr, program.ptr()));
    ASSERT_NE(program, nullptr);
    ASSERT_SUCCESS(urProgramBuild(context, program, nullptr));

    uur::raii::Kernel kernel = nullptr;
    ASSERT_SUCCESS(urKernelCreate(program, "_Z8myKernelPi", kernel.ptr()));
    ASSERT_NE(kernel, nullptr);

    int number = 10;
    ASSERT_SUCCESS(
        urKernelSetArgValue(kernel, 0, sizeof(int), nullptr, &number));
    const auto &kernelArgs = kernel->getArgIndices();
    ASSERT_EQ(kernelArgs.size(), 1 + NumberOfImplicitArgsCUDA);

    int storedValue = *static_cast<const int *>(kernelArgs[0]);
    ASSERT_EQ(storedValue, number);
}

TEST_P(cudaKernelTest, URKernelArgumentSetTwice) {
    uur::raii::Program program = nullptr;
    ASSERT_SUCCESS(urProgramCreateWithBinary(
        context, device, std::strlen(ptxSource), (const uint8_t *)ptxSource,
        nullptr, program.ptr()));
    ASSERT_NE(program, nullptr);
    ASSERT_SUCCESS(urProgramBuild(context, program, nullptr));

    uur::raii::Kernel kernel = nullptr;
    ASSERT_SUCCESS(urKernelCreate(program, "_Z8myKernelPi", kernel.ptr()));
    ASSERT_NE(kernel, nullptr);

    int number = 10;
    ASSERT_SUCCESS(
        urKernelSetArgValue(kernel, 0, sizeof(int), nullptr, &number));
    const auto &kernelArgs = kernel->getArgIndices();
    ASSERT_EQ(kernelArgs.size(), 1 + NumberOfImplicitArgsCUDA);
    int storedValue = *static_cast<const int *>(kernelArgs[0]);
    ASSERT_EQ(storedValue, number);

    int otherNumber = 934;
    ASSERT_SUCCESS(
        urKernelSetArgValue(kernel, 0, sizeof(int), nullptr, &otherNumber));
    const auto kernelArgs2 = kernel->getArgIndices();
    ASSERT_EQ(kernelArgs2.size(), 1 + NumberOfImplicitArgsCUDA);
    storedValue = *static_cast<const int *>(kernelArgs2[0]);
    ASSERT_EQ(storedValue, otherNumber);
}

TEST_P(cudaKernelTest, URKernelDispatch) {
    uur::raii::Program program = nullptr;
    ASSERT_SUCCESS(urProgramCreateWithBinary(
        context, device, std::strlen(ptxSource), (const uint8_t *)ptxSource,
        nullptr, program.ptr()));
    ASSERT_NE(program, nullptr);
    ASSERT_SUCCESS(urProgramBuild(context, program, nullptr));

    uur::raii::Kernel kernel = nullptr;
    ASSERT_SUCCESS(urKernelCreate(program, "_Z8myKernelPi", kernel.ptr()));
    ASSERT_NE(kernel, nullptr);

    const size_t memSize = 1024u;
    uur::raii::Mem buffer = nullptr;
    ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE, memSize,
                                     nullptr, buffer.ptr()));
    ASSERT_NE(buffer, nullptr);
    ASSERT_SUCCESS(urKernelSetArgMemObj(kernel, 0, nullptr, buffer));

    const size_t workDim = 1;
    const size_t globalWorkOffset[] = {0};
    const size_t globalWorkSize[] = {1};
    const size_t localWorkSize[] = {1};
    ASSERT_SUCCESS(urEnqueueKernelLaunch(queue, kernel, workDim,
                                         globalWorkOffset, globalWorkSize,
                                         localWorkSize, 0, nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));
}

TEST_P(cudaKernelTest, URKernelDispatchTwo) {
    uur::raii::Program program = nullptr;
    ASSERT_SUCCESS(urProgramCreateWithBinary(
        context, device, std::strlen(ptxSource), (const uint8_t *)twoParams,
        nullptr, program.ptr()));
    ASSERT_NE(program, nullptr);
    ASSERT_SUCCESS(urProgramBuild(context, program, nullptr));

    uur::raii::Kernel kernel = nullptr;
    ASSERT_SUCCESS(urKernelCreate(program, "twoParamKernel", kernel.ptr()));
    ASSERT_NE(kernel, nullptr);

    const size_t memSize = 1024u;
    uur::raii::Mem buffer1 = nullptr;
    uur::raii::Mem buffer2 = nullptr;
    ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE, memSize,
                                     nullptr, buffer1.ptr()));
    ASSERT_NE(buffer1, nullptr);
    ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE, memSize,
                                     nullptr, buffer2.ptr()));
    ASSERT_NE(buffer1, nullptr);
    ASSERT_SUCCESS(urKernelSetArgMemObj(kernel, 0, nullptr, buffer1));
    ASSERT_SUCCESS(urKernelSetArgMemObj(kernel, 1, nullptr, buffer2));

    const size_t workDim = 1;
    const size_t globalWorkOffset[] = {0};
    const size_t globalWorkSize[] = {1};
    const size_t localWorkSize[] = {1};
    ASSERT_SUCCESS(urEnqueueKernelLaunch(queue, kernel, workDim,
                                         globalWorkOffset, globalWorkSize,
                                         localWorkSize, 0, nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));
}
