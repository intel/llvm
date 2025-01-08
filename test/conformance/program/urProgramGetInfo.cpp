// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>
#include <uur/known_failure.h>

struct urProgramGetInfoTest : uur::urProgramTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urProgramTest::SetUp());
        // Some queries need the program to be built.
        ASSERT_SUCCESS(urProgramBuild(this->context, program, nullptr));
    }
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urProgramGetInfoTest);

TEST_P(urProgramGetInfoTest, SuccessReferenceCount) {
    size_t size = 0;
    auto info_type = UR_PROGRAM_INFO_REFERENCE_COUNT;
    ASSERT_SUCCESS(urProgramGetInfo(program, info_type, 0, nullptr, &size));
    ASSERT_EQ(size, sizeof(uint32_t));

    uint32_t returned_reference_count = 0;
    ASSERT_SUCCESS(urProgramGetInfo(program, info_type, size,
                                    &returned_reference_count, nullptr));

    ASSERT_GT(returned_reference_count, 0U);
}

TEST_P(urProgramGetInfoTest, SuccessContext) {
    size_t size = 0;
    auto info_type = UR_PROGRAM_INFO_CONTEXT;
    ASSERT_SUCCESS(urProgramGetInfo(program, info_type, 0, nullptr, &size));
    ASSERT_EQ(size, sizeof(ur_context_handle_t));

    ur_context_handle_t returned_context = nullptr;
    ASSERT_SUCCESS(
        urProgramGetInfo(program, info_type, size, &returned_context, nullptr));

    ASSERT_EQ(returned_context, context);
}

TEST_P(urProgramGetInfoTest, SuccessNumDevices) {
    size_t size = 0;
    auto info_type = UR_PROGRAM_INFO_NUM_DEVICES;
    ASSERT_SUCCESS(urProgramGetInfo(program, info_type, 0, nullptr, &size));
    ASSERT_EQ(size, sizeof(uint32_t));

    uint32_t returned_num_devices = 0;
    ASSERT_SUCCESS(urProgramGetInfo(program, info_type, size,
                                    &returned_num_devices, nullptr));

    ASSERT_GE(uur::DevicesEnvironment::instance->devices.size(),
              returned_num_devices);
}

TEST_P(urProgramGetInfoTest, SuccessDevices) {
    size_t size = 0;
    auto info_type = UR_PROGRAM_INFO_DEVICES;
    ASSERT_SUCCESS(urProgramGetInfo(program, info_type, 0, nullptr, &size));
    ASSERT_EQ(size, sizeof(ur_context_handle_t));

    std::vector<ur_device_handle_t> returned_devices(size);
    ASSERT_SUCCESS(urProgramGetInfo(program, info_type, size,
                                    returned_devices.data(), nullptr));

    size_t devices_count = size / sizeof(ur_device_handle_t);

    ASSERT_EQ(devices_count, 1);
    ASSERT_EQ(returned_devices[0], device);
}

TEST_P(urProgramGetInfoTest, SuccessIL) {
    size_t size = 0;
    auto info_type = UR_PROGRAM_INFO_IL;
    ASSERT_SUCCESS(urProgramGetInfo(program, info_type, 0, nullptr, &size));
    ASSERT_GE(size, 0);

    std::vector<char> returned_il(size);
    // Some adapters only support ProgramCreateWithBinary, in those cases we
    // expect a return size of 0 and an empty return value for INFO_IL.
    if (size > 0) {
        ASSERT_SUCCESS(urProgramGetInfo(program, UR_PROGRAM_INFO_IL, size,
                                        returned_il.data(), nullptr));
        ASSERT_EQ(returned_il, *il_binary.get());
    } else {
        ASSERT_TRUE(returned_il.empty());
    }
}

TEST_P(urProgramGetInfoTest, SuccessBinarySizes) {
    size_t size = 0;
    auto info_type = UR_PROGRAM_INFO_BINARY_SIZES;
    ASSERT_SUCCESS(urProgramGetInfo(program, info_type, 0, nullptr, &size));
    ASSERT_NE(size, 0);

    std::vector<size_t> binary_sizes(size / sizeof(size_t));
    ASSERT_SUCCESS(urProgramGetInfo(program, UR_PROGRAM_INFO_BINARY_SIZES, size,
                                    binary_sizes.data(), nullptr));

    for (const auto &binary_size : binary_sizes) {
        ASSERT_GT(binary_size, 0);
    }
}

TEST_P(urProgramGetInfoTest, SuccessBinaries) {
    size_t binary_sizes_len = 0;
    std::vector<char> property_value;
    ASSERT_SUCCESS(urProgramGetInfo(program, UR_PROGRAM_INFO_BINARY_SIZES, 0,
                                    nullptr, &binary_sizes_len));
    // Due to how the fixtures + env are set up we should only have one
    // device associated with program, so one binary.
    ASSERT_EQ(binary_sizes_len / sizeof(size_t), 1);

    size_t binary_sizes[1] = {binary_sizes_len};
    ASSERT_SUCCESS(urProgramGetInfo(program, UR_PROGRAM_INFO_BINARY_SIZES,
                                    binary_sizes_len, binary_sizes, nullptr));
    property_value.resize(binary_sizes[0]);
    char *binaries[1] = {property_value.data()};
    ASSERT_SUCCESS(urProgramGetInfo(program, UR_PROGRAM_INFO_BINARIES,
                                    sizeof(binaries[0]), binaries, nullptr));
}

TEST_P(urProgramGetInfoTest, SuccessNumKernels) {
    UUR_KNOWN_FAILURE_ON(uur::HIP{});

    size_t size = 0;
    auto info_type = UR_PROGRAM_INFO_NUM_KERNELS;
    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urProgramGetInfo(program, info_type, 0, nullptr, &size), info_type);
    ASSERT_EQ(size, sizeof(size_t));

    size_t returned_num_kernels = 0;
    ASSERT_SUCCESS(urProgramGetInfo(program, info_type, size,
                                    &returned_num_kernels, nullptr));

    ASSERT_GT(returned_num_kernels, 0U);
}

TEST_P(urProgramGetInfoTest, SuccessKernelNames) {
    size_t size = 0;
    auto info_type = UR_PROGRAM_INFO_KERNEL_NAMES;
    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urProgramGetInfo(program, info_type, 0, nullptr, &size), info_type);
    ASSERT_GT(size, 0);

    std::vector<char> returned_kernel_names(size);
    returned_kernel_names[size - 1] = 'x';
    ASSERT_SUCCESS(urProgramGetInfo(program, info_type, size,
                                    returned_kernel_names.data(), nullptr));

    ASSERT_EQ(size, returned_kernel_names.size());
    ASSERT_EQ(returned_kernel_names[size - 1], '\0');
}

TEST_P(urProgramGetInfoTest, InvalidNullHandleProgram) {
    uint32_t ref_count = 0;
    ASSERT_EQ_RESULT(urProgramGetInfo(nullptr, UR_PROGRAM_INFO_REFERENCE_COUNT,
                                      sizeof(ref_count), &ref_count, nullptr),
                     UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urProgramGetInfoTest, InvalidEnumeration) {
    size_t prop_size = 0;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                     urProgramGetInfo(program, UR_PROGRAM_INFO_FORCE_UINT32, 0,
                                      nullptr, &prop_size));
}

TEST_P(urProgramGetInfoTest, InvalidSizeZero) {
    uint32_t ref_count = 0;
    ASSERT_EQ_RESULT(urProgramGetInfo(program, UR_PROGRAM_INFO_REFERENCE_COUNT,
                                      0, &ref_count, nullptr),
                     UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urProgramGetInfoTest, InvalidSizeSmall) {
    uint32_t ref_count = 0;
    ASSERT_EQ_RESULT(urProgramGetInfo(program, UR_PROGRAM_INFO_REFERENCE_COUNT,
                                      sizeof(ref_count) - 1, &ref_count,
                                      nullptr),
                     UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urProgramGetInfoTest, InvalidNullPointerPropValue) {
    ASSERT_EQ_RESULT(urProgramGetInfo(program, UR_PROGRAM_INFO_REFERENCE_COUNT,
                                      sizeof(uint32_t), nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urProgramGetInfoTest, InvalidNullPointerPropValueRet) {
    ASSERT_EQ_RESULT(urProgramGetInfo(program, UR_PROGRAM_INFO_REFERENCE_COUNT,
                                      0, nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urProgramGetInfoTest, NumDevicesIsNonzero) {
    uint32_t count = 0;
    ASSERT_SUCCESS(urProgramGetInfo(program, UR_PROGRAM_INFO_NUM_DEVICES,
                                    sizeof(uint32_t), &count, nullptr));
    ASSERT_GE(count, 1);
}

TEST_P(urProgramGetInfoTest, NumDevicesMatchesDeviceArray) {
    uint32_t count = 0;
    ASSERT_SUCCESS(urProgramGetInfo(program, UR_PROGRAM_INFO_NUM_DEVICES,
                                    sizeof(uint32_t), &count, nullptr));

    size_t info_devices_size = 0;
    ASSERT_SUCCESS(urProgramGetInfo(program, UR_PROGRAM_INFO_DEVICES, 0,
                                    nullptr, &info_devices_size));
    ASSERT_EQ(count, info_devices_size / sizeof(ur_device_handle_t));
}

TEST_P(urProgramGetInfoTest, NumDevicesMatchesContextNumDevices) {
    uint32_t count = 0;
    ASSERT_SUCCESS(urProgramGetInfo(program, UR_PROGRAM_INFO_NUM_DEVICES,
                                    sizeof(uint32_t), &count, nullptr));

    // The device count either matches the number of devices in the context or
    // is 1, depending on how it was built
    uint32_t info_context_devices_count = 0;
    ASSERT_SUCCESS(urContextGetInfo(context, UR_CONTEXT_INFO_NUM_DEVICES,
                                    sizeof(uint32_t),
                                    &info_context_devices_count, nullptr));
    ASSERT_TRUE(count == 1 || count == info_context_devices_count);
}
