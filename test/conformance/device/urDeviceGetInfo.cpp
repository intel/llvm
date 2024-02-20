// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <map>
#include <uur/fixtures.h>

static std::unordered_map<ur_device_info_t, size_t> device_info_size_map = {
    {UR_DEVICE_INFO_TYPE, sizeof(ur_device_type_t)},
    {UR_DEVICE_INFO_VENDOR_ID, sizeof(uint32_t)},
    {UR_DEVICE_INFO_DEVICE_ID, sizeof(uint32_t)},
    {UR_DEVICE_INFO_MAX_COMPUTE_UNITS, sizeof(uint32_t)},
    {UR_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS, sizeof(uint32_t)},
    {UR_DEVICE_INFO_MAX_WORK_GROUP_SIZE, sizeof(size_t)},
    {UR_DEVICE_INFO_SINGLE_FP_CONFIG, sizeof(ur_device_fp_capability_flags_t)},
    {UR_DEVICE_INFO_HALF_FP_CONFIG, sizeof(ur_device_fp_capability_flags_t)},
    {UR_DEVICE_INFO_DOUBLE_FP_CONFIG, sizeof(ur_device_fp_capability_flags_t)},
    {UR_DEVICE_INFO_QUEUE_PROPERTIES, sizeof(ur_queue_flags_t)},
    {UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_CHAR, sizeof(uint32_t)},
    {UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_SHORT, sizeof(uint32_t)},
    {UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_INT, sizeof(uint32_t)},
    {UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_LONG, sizeof(uint32_t)},
    {UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof(uint32_t)},
    {UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_DOUBLE, sizeof(uint32_t)},
    {UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_HALF, sizeof(uint32_t)},
    {UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR, sizeof(uint32_t)},
    {UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT, sizeof(uint32_t)},
    {UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT, sizeof(uint32_t)},
    {UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG, sizeof(uint32_t)},
    {UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT, sizeof(uint32_t)},
    {UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE, sizeof(uint32_t)},
    {UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF, sizeof(uint32_t)},
    {UR_DEVICE_INFO_MAX_CLOCK_FREQUENCY, sizeof(uint32_t)},
    {UR_DEVICE_INFO_MEMORY_CLOCK_RATE, sizeof(uint32_t)},
    {UR_DEVICE_INFO_ADDRESS_BITS, sizeof(uint32_t)},
    {UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE, sizeof(uint64_t)},
    {UR_DEVICE_INFO_IMAGE_SUPPORTED, sizeof(ur_bool_t)},
    {UR_DEVICE_INFO_MAX_READ_IMAGE_ARGS, sizeof(uint32_t)},
    {UR_DEVICE_INFO_MAX_WRITE_IMAGE_ARGS, sizeof(uint32_t)},
    {UR_DEVICE_INFO_MAX_READ_WRITE_IMAGE_ARGS, sizeof(uint32_t)},
    {UR_DEVICE_INFO_IMAGE2D_MAX_WIDTH, sizeof(size_t)},
    {UR_DEVICE_INFO_IMAGE2D_MAX_HEIGHT, sizeof(size_t)},
    {UR_DEVICE_INFO_IMAGE3D_MAX_WIDTH, sizeof(size_t)},
    {UR_DEVICE_INFO_IMAGE3D_MAX_HEIGHT, sizeof(size_t)},
    {UR_DEVICE_INFO_IMAGE3D_MAX_DEPTH, sizeof(size_t)},
    {UR_DEVICE_INFO_IMAGE_MAX_BUFFER_SIZE, sizeof(size_t)},
    {UR_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE, sizeof(size_t)},
    {UR_DEVICE_INFO_MAX_SAMPLERS, sizeof(uint32_t)},
    {UR_DEVICE_INFO_MAX_PARAMETER_SIZE, sizeof(size_t)},
    {UR_DEVICE_INFO_MEM_BASE_ADDR_ALIGN, sizeof(uint32_t)},
    {UR_DEVICE_INFO_GLOBAL_MEM_CACHE_TYPE, sizeof(ur_device_mem_cache_type_t)},
    {UR_DEVICE_INFO_GLOBAL_MEM_CACHELINE_SIZE, sizeof(uint32_t)},
    {UR_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE, sizeof(uint64_t)},
    {UR_DEVICE_INFO_GLOBAL_MEM_SIZE, sizeof(uint64_t)},
    {UR_DEVICE_INFO_GLOBAL_MEM_FREE, sizeof(uint64_t)},
    {UR_DEVICE_INFO_MAX_CONSTANT_BUFFER_SIZE, sizeof(uint64_t)},
    {UR_DEVICE_INFO_MAX_CONSTANT_ARGS, sizeof(uint32_t)},
    {UR_DEVICE_INFO_LOCAL_MEM_TYPE, sizeof(ur_device_local_mem_type_t)},
    {UR_DEVICE_INFO_LOCAL_MEM_SIZE, sizeof(uint64_t)},
    {UR_DEVICE_INFO_ERROR_CORRECTION_SUPPORT, sizeof(ur_bool_t)},
    {UR_DEVICE_INFO_HOST_UNIFIED_MEMORY, sizeof(ur_bool_t)},
    {UR_DEVICE_INFO_PROFILING_TIMER_RESOLUTION, sizeof(size_t)},
    {UR_DEVICE_INFO_ENDIAN_LITTLE, sizeof(ur_bool_t)},
    {UR_DEVICE_INFO_AVAILABLE, sizeof(ur_bool_t)},
    {UR_DEVICE_INFO_COMPILER_AVAILABLE, sizeof(ur_bool_t)},
    {UR_DEVICE_INFO_LINKER_AVAILABLE, sizeof(ur_bool_t)},
    {UR_DEVICE_INFO_EXECUTION_CAPABILITIES,
     sizeof(ur_device_exec_capability_flags_t)},
    {UR_DEVICE_INFO_QUEUE_ON_DEVICE_PROPERTIES, sizeof(ur_queue_flags_t)},
    {UR_DEVICE_INFO_QUEUE_ON_HOST_PROPERTIES, sizeof(ur_queue_flags_t)},
    {UR_DEVICE_INFO_PLATFORM, sizeof(ur_platform_handle_t)},
    {UR_DEVICE_INFO_REFERENCE_COUNT, sizeof(uint32_t)},
    {UR_DEVICE_INFO_PRINTF_BUFFER_SIZE, sizeof(size_t)},
    {UR_DEVICE_INFO_PREFERRED_INTEROP_USER_SYNC, sizeof(ur_bool_t)},
    {UR_DEVICE_INFO_PARENT_DEVICE, sizeof(ur_device_handle_t)},
    {UR_DEVICE_INFO_PARTITION_MAX_SUB_DEVICES, sizeof(uint32_t)},
    {UR_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN,
     sizeof(ur_device_affinity_domain_flags_t)},
    {UR_DEVICE_INFO_MAX_NUM_SUB_GROUPS, sizeof(uint32_t)},
    {UR_DEVICE_INFO_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS, sizeof(ur_bool_t)},
    {UR_DEVICE_INFO_USM_HOST_SUPPORT,
     sizeof(ur_device_usm_access_capability_flags_t)},
    {UR_DEVICE_INFO_USM_DEVICE_SUPPORT,
     sizeof(ur_device_usm_access_capability_flags_t)},
    {UR_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT,
     sizeof(ur_device_usm_access_capability_flags_t)},
    {UR_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT,
     sizeof(ur_device_usm_access_capability_flags_t)},
    {UR_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT,
     sizeof(ur_device_usm_access_capability_flags_t)},
    {UR_DEVICE_INFO_GPU_EU_COUNT, sizeof(uint32_t)},
    {UR_DEVICE_INFO_GPU_EU_SIMD_WIDTH, sizeof(uint32_t)},
    {UR_DEVICE_INFO_GPU_EU_SLICES, sizeof(uint32_t)},
    {UR_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE, sizeof(uint32_t)},
    {UR_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE, sizeof(uint32_t)},
    {UR_DEVICE_INFO_GPU_HW_THREADS_PER_EU, sizeof(uint32_t)},
    {UR_DEVICE_INFO_MAX_MEMORY_BANDWIDTH, sizeof(uint32_t)},
    {UR_DEVICE_INFO_IMAGE_SRGB, sizeof(ur_bool_t)},
    {UR_DEVICE_INFO_BUILD_ON_SUBDEVICE, sizeof(ur_bool_t)},
    {UR_DEVICE_INFO_ATOMIC_64, sizeof(ur_bool_t)},
    {UR_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES,
     sizeof(ur_memory_order_capability_flags_t)},
    {UR_DEVICE_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES,
     sizeof(ur_memory_scope_capability_flags_t)},
    {UR_DEVICE_INFO_BFLOAT16, sizeof(ur_bool_t)},
    {UR_DEVICE_INFO_MAX_COMPUTE_QUEUE_INDICES, sizeof(uint32_t)},
    {UR_DEVICE_INFO_KERNEL_SET_SPECIALIZATION_CONSTANTS, sizeof(ur_bool_t)},
    {UR_DEVICE_INFO_MEMORY_BUS_WIDTH, sizeof(uint32_t)},
    {UR_DEVICE_INFO_MAX_WORK_GROUPS_3D, sizeof(size_t[3])},
    {UR_DEVICE_INFO_ASYNC_BARRIER, sizeof(ur_bool_t)},
    {UR_DEVICE_INFO_MEM_CHANNEL_SUPPORT, sizeof(ur_bool_t)},
    {UR_DEVICE_INFO_HOST_PIPE_READ_WRITE_SUPPORTED, sizeof(ur_bool_t)},
    {UR_DEVICE_INFO_MAX_REGISTERS_PER_WORK_GROUP, sizeof(uint32_t)},
    {UR_DEVICE_INFO_COMPONENT_DEVICES, sizeof(uint32_t)},
    {UR_DEVICE_INFO_COMPOSITE_DEVICE, sizeof(ur_device_handle_t)},
};

struct urDeviceGetInfoTest : uur::urAllDevicesTest,
                             ::testing::WithParamInterface<ur_device_info_t> {

    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(uur::urAllDevicesTest::SetUp());
    }
};

INSTANTIATE_TEST_SUITE_P(
    , urDeviceGetInfoTest,
    ::testing::Values(

        UR_DEVICE_INFO_TYPE,
        UR_DEVICE_INFO_VENDOR_ID,                              //
        UR_DEVICE_INFO_DEVICE_ID,                              //
        UR_DEVICE_INFO_MAX_COMPUTE_UNITS,                      //
        UR_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS,               //
        UR_DEVICE_INFO_MAX_WORK_ITEM_SIZES,                    //
        UR_DEVICE_INFO_MAX_WORK_GROUP_SIZE,                    //
        UR_DEVICE_INFO_SINGLE_FP_CONFIG,                       //
        UR_DEVICE_INFO_HALF_FP_CONFIG,                         //
        UR_DEVICE_INFO_DOUBLE_FP_CONFIG,                       //
        UR_DEVICE_INFO_QUEUE_PROPERTIES,                       //
        UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_CHAR,            //
        UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_SHORT,           //
        UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_LONG,            //
        UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_FLOAT,           //
        UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR,               //
        UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT,              //
        UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT,                //
        UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG,               //
        UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT,              //
        UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE,             //
        UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF,               //
        UR_DEVICE_INFO_MAX_CLOCK_FREQUENCY,                    //
        UR_DEVICE_INFO_MEMORY_CLOCK_RATE,                      //
        UR_DEVICE_INFO_ADDRESS_BITS,                           //
        UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE,                     //
        UR_DEVICE_INFO_IMAGE_SUPPORTED,                        //
        UR_DEVICE_INFO_MAX_READ_IMAGE_ARGS,                    //
        UR_DEVICE_INFO_MAX_WRITE_IMAGE_ARGS,                   //
        UR_DEVICE_INFO_MAX_READ_WRITE_IMAGE_ARGS,              //
        UR_DEVICE_INFO_IMAGE2D_MAX_WIDTH,                      //
        UR_DEVICE_INFO_IMAGE2D_MAX_HEIGHT,                     //
        UR_DEVICE_INFO_IMAGE3D_MAX_WIDTH,                      //
        UR_DEVICE_INFO_IMAGE3D_MAX_HEIGHT,                     //
        UR_DEVICE_INFO_IMAGE3D_MAX_DEPTH,                      //
        UR_DEVICE_INFO_IMAGE_MAX_BUFFER_SIZE,                  //
        UR_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE,                   //
        UR_DEVICE_INFO_MAX_SAMPLERS,                           //
        UR_DEVICE_INFO_MAX_PARAMETER_SIZE,                     //
        UR_DEVICE_INFO_MEM_BASE_ADDR_ALIGN,                    //
        UR_DEVICE_INFO_GLOBAL_MEM_CACHE_TYPE,                  //
        UR_DEVICE_INFO_GLOBAL_MEM_CACHELINE_SIZE,              //
        UR_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE,                  //
        UR_DEVICE_INFO_GLOBAL_MEM_SIZE,                        //
        UR_DEVICE_INFO_GLOBAL_MEM_FREE,                        //
        UR_DEVICE_INFO_MAX_CONSTANT_BUFFER_SIZE,               //
        UR_DEVICE_INFO_MAX_CONSTANT_ARGS,                      //
        UR_DEVICE_INFO_LOCAL_MEM_TYPE,                         //
        UR_DEVICE_INFO_LOCAL_MEM_SIZE,                         //
        UR_DEVICE_INFO_ERROR_CORRECTION_SUPPORT,               //
        UR_DEVICE_INFO_HOST_UNIFIED_MEMORY,                    //
        UR_DEVICE_INFO_PROFILING_TIMER_RESOLUTION,             //
        UR_DEVICE_INFO_ENDIAN_LITTLE,                          //
        UR_DEVICE_INFO_AVAILABLE,                              //
        UR_DEVICE_INFO_COMPILER_AVAILABLE,                     //
        UR_DEVICE_INFO_LINKER_AVAILABLE,                       //
        UR_DEVICE_INFO_EXECUTION_CAPABILITIES,                 //
        UR_DEVICE_INFO_QUEUE_ON_DEVICE_PROPERTIES,             //
        UR_DEVICE_INFO_QUEUE_ON_HOST_PROPERTIES,               //
        UR_DEVICE_INFO_BUILT_IN_KERNELS,                       //
        UR_DEVICE_INFO_PLATFORM,                               //
        UR_DEVICE_INFO_REFERENCE_COUNT,                        //
        UR_DEVICE_INFO_IL_VERSION,                             //
        UR_DEVICE_INFO_NAME,                                   //
        UR_DEVICE_INFO_VENDOR,                                 //
        UR_DEVICE_INFO_DRIVER_VERSION,                         //
        UR_DEVICE_INFO_PROFILE,                                //
        UR_DEVICE_INFO_VERSION,                                //
        UR_DEVICE_INFO_BACKEND_RUNTIME_VERSION,                //
        UR_DEVICE_INFO_EXTENSIONS,                             //
        UR_DEVICE_INFO_PRINTF_BUFFER_SIZE,                     //
        UR_DEVICE_INFO_PREFERRED_INTEROP_USER_SYNC,            //
        UR_DEVICE_INFO_PARENT_DEVICE,                          //
        UR_DEVICE_INFO_SUPPORTED_PARTITIONS,                   //
        UR_DEVICE_INFO_PARTITION_MAX_SUB_DEVICES,              //
        UR_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN,              //
        UR_DEVICE_INFO_PARTITION_TYPE,                         //
        UR_DEVICE_INFO_MAX_NUM_SUB_GROUPS,                     //
        UR_DEVICE_INFO_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS, //
        UR_DEVICE_INFO_SUB_GROUP_SIZES_INTEL,                  //
        UR_DEVICE_INFO_USM_HOST_SUPPORT,                       //
        UR_DEVICE_INFO_USM_DEVICE_SUPPORT,                     //
        UR_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT,              //
        UR_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT,               //
        UR_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT,              //
        UR_DEVICE_INFO_UUID,                                   //
        UR_DEVICE_INFO_PCI_ADDRESS,                            //
        UR_DEVICE_INFO_GPU_EU_COUNT,                           //
        UR_DEVICE_INFO_GPU_EU_SIMD_WIDTH,                      //
        UR_DEVICE_INFO_GPU_EU_SLICES,                          //
        UR_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE,              //
        UR_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE,                //
        UR_DEVICE_INFO_GPU_HW_THREADS_PER_EU,                  //
        UR_DEVICE_INFO_MAX_MEMORY_BANDWIDTH,                   //
        UR_DEVICE_INFO_IMAGE_SRGB,                             //
        UR_DEVICE_INFO_BUILD_ON_SUBDEVICE,                     //
        UR_DEVICE_INFO_ATOMIC_64,                              //
        UR_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES,       //
        UR_DEVICE_INFO_BFLOAT16,                               //
        UR_DEVICE_INFO_MAX_COMPUTE_QUEUE_INDICES,              //
        UR_DEVICE_INFO_MEMORY_BUS_WIDTH,                       //
        UR_DEVICE_INFO_MAX_WORK_GROUPS_3D,                     //
        UR_DEVICE_INFO_ASYNC_BARRIER,                          //
        UR_DEVICE_INFO_MEM_CHANNEL_SUPPORT,                    //
        UR_DEVICE_INFO_HOST_PIPE_READ_WRITE_SUPPORTED,         //
        UR_DEVICE_INFO_MAX_REGISTERS_PER_WORK_GROUP,           //
        UR_DEVICE_INFO_VIRTUAL_MEMORY_SUPPORT                  //
        ),
    [](const ::testing::TestParamInfo<ur_device_info_t> &info) {
        std::stringstream ss;
        ss << info.param;
        return ss.str();
    });

bool doesReturnArray(ur_device_info_t info_type) {
    if (info_type == UR_DEVICE_INFO_SUPPORTED_PARTITIONS ||
        info_type == UR_DEVICE_INFO_PARTITION_TYPE) {
        return true;
    }
    return false;
}

TEST_P(urDeviceGetInfoTest, Success) {
    ur_device_info_t info_type = GetParam();
    for (auto device : devices) {
        size_t size = 0;
        ur_result_t result =
            urDeviceGetInfo(device, info_type, 0, nullptr, &size);

        if (result == UR_RESULT_SUCCESS) {
            if (doesReturnArray(info_type) && size == 0) {
                return;
            }
            ASSERT_NE(size, 0);

            if (const auto expected_size = device_info_size_map.find(info_type);
                expected_size != device_info_size_map.end()) {
                ASSERT_EQ(expected_size->second, size);
            }

            std::vector<char> info_data(size);
            ASSERT_SUCCESS(urDeviceGetInfo(device, info_type, size,
                                           info_data.data(), nullptr));
        } else {
            ASSERT_EQ_RESULT(result, UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION);
        }
    }
}

TEST_P(urDeviceGetInfoTest, InvalidNullHandleDevice) {
    ur_device_type_t device_type;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urDeviceGetInfo(nullptr, UR_DEVICE_INFO_TYPE,
                                     sizeof(ur_device_type_t), &device_type,
                                     nullptr));
}

TEST_P(urDeviceGetInfoTest, InvalidEnumerationInfoType) {
    for (auto device : devices) {
        ur_device_type_t device_type;
        ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                         urDeviceGetInfo(device, UR_DEVICE_INFO_FORCE_UINT32,
                                         sizeof(ur_device_type_t), &device_type,
                                         nullptr));
    }
}

TEST_P(urDeviceGetInfoTest, InvalidSizePropSize) {
    for (auto device : devices) {
        ur_device_type_t device_type;
        ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                         urDeviceGetInfo(device, UR_DEVICE_INFO_TYPE, 0,
                                         &device_type, nullptr));
    }
}

TEST_P(urDeviceGetInfoTest, InvalidSizePropSizeSmall) {
    for (auto device : devices) {
        ur_device_type_t device_type;
        ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                         urDeviceGetInfo(device, UR_DEVICE_INFO_TYPE,
                                         sizeof(device_type) - 1, &device_type,
                                         nullptr));
    }
}

TEST_P(urDeviceGetInfoTest, InvalidNullPointerPropValue) {
    for (auto device : devices) {
        ur_device_type_t device_type;
        ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                         urDeviceGetInfo(device, UR_DEVICE_INFO_TYPE,
                                         sizeof(device_type), nullptr,
                                         nullptr));
    }
}

TEST_P(urDeviceGetInfoTest, InvalidNullPointerPropSizeRet) {
    for (auto device : devices) {
        ASSERT_EQ_RESULT(
            UR_RESULT_ERROR_INVALID_NULL_POINTER,
            urDeviceGetInfo(device, UR_DEVICE_INFO_TYPE, 0, nullptr, nullptr));
    }
}
