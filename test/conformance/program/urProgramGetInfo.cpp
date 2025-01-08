// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

struct urProgramGetInfoTest : uur::urProgramTestWithParam<ur_program_info_t> {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(
            urProgramTestWithParam<ur_program_info_t>::SetUp());
        // Some queries need the program to be built.
        ASSERT_SUCCESS(urProgramBuild(this->context, program, nullptr));
    }
};

UUR_TEST_SUITE_P(
    urProgramGetInfoTest,
    ::testing::Values(UR_PROGRAM_INFO_REFERENCE_COUNT, UR_PROGRAM_INFO_CONTEXT,
                      UR_PROGRAM_INFO_NUM_DEVICES, UR_PROGRAM_INFO_DEVICES,
                      UR_PROGRAM_INFO_IL, UR_PROGRAM_INFO_BINARY_SIZES,
                      UR_PROGRAM_INFO_BINARIES, UR_PROGRAM_INFO_NUM_KERNELS,
                      UR_PROGRAM_INFO_KERNEL_NAMES),
    uur::deviceTestWithParamPrinter<ur_program_info_t>);

struct urProgramGetInfoSingleTest : uur::urProgramTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urProgramTest::SetUp());
        ASSERT_SUCCESS(urProgramBuild(this->context, program, nullptr));
    }
};
UUR_INSTANTIATE_KERNEL_TEST_SUITE_P(urProgramGetInfoSingleTest);

TEST_P(urProgramGetInfoTest, Success) {
    auto property_name = getParam();
    std::vector<char> property_value;
    size_t property_size = 0;
    if (property_name == UR_PROGRAM_INFO_BINARIES) {
        size_t binary_sizes_len = 0;
        ASSERT_SUCCESS(urProgramGetInfo(program, UR_PROGRAM_INFO_BINARY_SIZES,
                                        0, nullptr, &binary_sizes_len));
        // Due to how the fixtures + env are set up we should only have one
        // device associated with program, so one binary.
        ASSERT_EQ(binary_sizes_len / sizeof(size_t), 1);
        size_t binary_sizes[1] = {binary_sizes_len};
        ASSERT_SUCCESS(urProgramGetInfo(program, UR_PROGRAM_INFO_BINARY_SIZES,
                                        binary_sizes_len, binary_sizes,
                                        nullptr));
        property_value.resize(binary_sizes[0]);
        char *binaries[1] = {property_value.data()};
        ASSERT_SUCCESS(urProgramGetInfo(program, UR_PROGRAM_INFO_BINARIES,
                                        sizeof(binaries[0]), binaries,
                                        nullptr));
    } else {
        ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
            urProgramGetInfo(program, property_name, 0, nullptr,
                             &property_size),
            property_name);
        if (property_size) {
            property_value.resize(property_size);
            ASSERT_SUCCESS(urProgramGetInfo(program, property_name,
                                            property_size,
                                            property_value.data(), nullptr));
        } else {
            ASSERT_EQ(property_name, UR_PROGRAM_INFO_IL);
        }
    }
    switch (property_name) {
    case UR_PROGRAM_INFO_REFERENCE_COUNT: {
        auto returned_reference_count =
            reinterpret_cast<uint32_t *>(property_value.data());
        ASSERT_GT(*returned_reference_count, 0U);
        break;
    }
    case UR_PROGRAM_INFO_CONTEXT: {
        auto returned_context =
            reinterpret_cast<ur_context_handle_t *>(property_value.data());
        ASSERT_EQ(context, *returned_context);
        break;
    }
    case UR_PROGRAM_INFO_NUM_DEVICES: {
        auto returned_num_of_devices =
            reinterpret_cast<uint32_t *>(property_value.data());
        ASSERT_GE(uur::DevicesEnvironment::instance->devices.size(),
                  *returned_num_of_devices);
        break;
    }
    case UR_PROGRAM_INFO_DEVICES: {
        auto returned_devices =
            reinterpret_cast<ur_device_handle_t *>(property_value.data());
        size_t devices_count = property_size / sizeof(ur_device_handle_t);
        ASSERT_GT(devices_count, 0);
        for (uint32_t i = 0; i < devices_count; i++) {
            auto &devices = uur::DevicesEnvironment::instance->devices;
            auto queried_device =
                std::find(devices.begin(), devices.end(), returned_devices[i]);
            EXPECT_TRUE(queried_device != devices.end());
        }
        break;
    }
    case UR_PROGRAM_INFO_NUM_KERNELS: {
        auto returned_num_of_kernels =
            reinterpret_cast<uint32_t *>(property_value.data());
        ASSERT_GT(*returned_num_of_kernels, 0U);
        break;
    }
    case UR_PROGRAM_INFO_KERNEL_NAMES: {
        auto returned_kernel_names =
            reinterpret_cast<char *>(property_value.data());
        ASSERT_STRNE(returned_kernel_names, "");
        break;
    }
    case UR_PROGRAM_INFO_IL: {
        // Some adapters only support ProgramCreateWithBinary, in those cases we
        // expect a return size of 0 and an empty return value for INFO_IL.
        if (!property_value.empty()) {
            ASSERT_EQ(property_value, *il_binary.get());
        }
        break;
    }
    default:
        break;
    }
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

TEST_P(urProgramGetInfoSingleTest, NumDevicesIsNonzero) {
    uint32_t count;
    ASSERT_SUCCESS(urProgramGetInfo(program, UR_PROGRAM_INFO_NUM_DEVICES,
                                    sizeof(uint32_t), &count, nullptr));
    ASSERT_GE(count, 1);
}

TEST_P(urProgramGetInfoSingleTest, NumDevicesMatchesDeviceArray) {
    uint32_t count;
    ASSERT_SUCCESS(urProgramGetInfo(program, UR_PROGRAM_INFO_NUM_DEVICES,
                                    sizeof(uint32_t), &count, nullptr));

    size_t info_devices_size;
    ASSERT_SUCCESS(urProgramGetInfo(program, UR_PROGRAM_INFO_DEVICES, 0,
                                    nullptr, &info_devices_size));
    ASSERT_EQ(count, info_devices_size / sizeof(ur_device_handle_t));
}

TEST_P(urProgramGetInfoSingleTest, NumDevicesMatchesContextNumDevices) {
    uint32_t count;
    ASSERT_SUCCESS(urProgramGetInfo(program, UR_PROGRAM_INFO_NUM_DEVICES,
                                    sizeof(uint32_t), &count, nullptr));

    // The device count either matches the number of devices in the context or
    // is 1, depending on how it was built
    uint32_t info_context_devices_count;
    ASSERT_SUCCESS(urContextGetInfo(context, UR_CONTEXT_INFO_NUM_DEVICES,
                                    sizeof(uint32_t),
                                    &info_context_devices_count, nullptr));
    ASSERT_TRUE(count == 1 || count == info_context_devices_count);
}
