// # curl https://raw.githubusercontent.com/oneapi-src/unified-runtime/refs/heads/main/scripts/core/device.yml 2>/dev/null \
// #  | awk '/name: .x_device_info_t/{flag=1} /^---/{flag=0} flag'  \
// #  | python a.py > ur_device_info_ret_types.inc
//
// import yaml
// import re
// import sys
//
// def get_type(x):
//     tags = x["desc"].split(" ")[0]
//     type = re.sub(r'\]\[.*', "]", tags)[1:-1]
//     type = type.replace("char[]", "std::string")
//     type = re.sub(r'(.*)\[\]', r"std::vector<\1>", type)
//     type = re.sub(r'(.*)\[(\d)\]', r"std::array<\1, \2>", type)
//     return type.replace("$x", "ur")
//
// for x in yaml.safe_load(sys.stdin)["etors"]:
//     print(f'MAP(UR_DEVICE_INFO_{x["name"]}, {get_type(x)})')
//
// With some manual post-processing

// TODO: Maybe parsing ur_api.h could be better, but YAML seems easier to
// process.

// clang-format off
MAP(UR_DEVICE_INFO_TYPE, ur_device_type_t)
MAP(UR_DEVICE_INFO_VENDOR_ID, uint32_t)
MAP(UR_DEVICE_INFO_DEVICE_ID, uint32_t)
MAP(UR_DEVICE_INFO_MAX_COMPUTE_UNITS, uint32_t)
MAP(UR_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS, uint32_t)
// Manually changed std::vector<size_t> -> std::array<size_t, 3>
MAP(UR_DEVICE_INFO_MAX_WORK_ITEM_SIZES, std::array<size_t, 3>)
MAP(UR_DEVICE_INFO_MAX_WORK_GROUP_SIZE, size_t)
MAP(UR_DEVICE_INFO_SINGLE_FP_CONFIG, ur_device_fp_capability_flags_t)
MAP(UR_DEVICE_INFO_HALF_FP_CONFIG, ur_device_fp_capability_flags_t)
MAP(UR_DEVICE_INFO_DOUBLE_FP_CONFIG, ur_device_fp_capability_flags_t)
MAP(UR_DEVICE_INFO_QUEUE_PROPERTIES, ur_queue_flags_t)
MAP(UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_CHAR, uint32_t)
MAP(UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_SHORT, uint32_t)
MAP(UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_INT, uint32_t)
MAP(UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_LONG, uint32_t)
MAP(UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_FLOAT, uint32_t)
MAP(UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_DOUBLE, uint32_t)
MAP(UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_HALF, uint32_t)
MAP(UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR, uint32_t)
MAP(UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT, uint32_t)
MAP(UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT, uint32_t)
MAP(UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG, uint32_t)
MAP(UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT, uint32_t)
MAP(UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE, uint32_t)
MAP(UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF, uint32_t)
MAP(UR_DEVICE_INFO_MAX_CLOCK_FREQUENCY, uint32_t)
MAP(UR_DEVICE_INFO_MEMORY_CLOCK_RATE, uint32_t)
MAP(UR_DEVICE_INFO_ADDRESS_BITS, uint32_t)
MAP(UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE, uint64_t)
MAP(UR_DEVICE_INFO_IMAGE_SUPPORT, ur_bool_t)
MAP(UR_DEVICE_INFO_MAX_READ_IMAGE_ARGS, uint32_t)
MAP(UR_DEVICE_INFO_MAX_WRITE_IMAGE_ARGS, uint32_t)
MAP(UR_DEVICE_INFO_MAX_READ_WRITE_IMAGE_ARGS, uint32_t)
MAP(UR_DEVICE_INFO_IMAGE2D_MAX_WIDTH, size_t)
MAP(UR_DEVICE_INFO_IMAGE2D_MAX_HEIGHT, size_t)
MAP(UR_DEVICE_INFO_IMAGE3D_MAX_WIDTH, size_t)
MAP(UR_DEVICE_INFO_IMAGE3D_MAX_HEIGHT, size_t)
MAP(UR_DEVICE_INFO_IMAGE3D_MAX_DEPTH, size_t)
MAP(UR_DEVICE_INFO_IMAGE_MAX_BUFFER_SIZE, size_t)
MAP(UR_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE, size_t)
MAP(UR_DEVICE_INFO_MAX_SAMPLERS, uint32_t)
MAP(UR_DEVICE_INFO_MAX_PARAMETER_SIZE, size_t)
MAP(UR_DEVICE_INFO_MEM_BASE_ADDR_ALIGN, uint32_t)
MAP(UR_DEVICE_INFO_GLOBAL_MEM_CACHE_TYPE, ur_device_mem_cache_type_t)
MAP(UR_DEVICE_INFO_GLOBAL_MEM_CACHELINE_SIZE, uint32_t)
MAP(UR_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE, uint64_t)
MAP(UR_DEVICE_INFO_GLOBAL_MEM_SIZE, uint64_t)
MAP(UR_DEVICE_INFO_GLOBAL_MEM_FREE, uint64_t)
MAP(UR_DEVICE_INFO_MAX_CONSTANT_BUFFER_SIZE, uint64_t)
MAP(UR_DEVICE_INFO_MAX_CONSTANT_ARGS, uint32_t)
MAP(UR_DEVICE_INFO_LOCAL_MEM_TYPE, ur_device_local_mem_type_t)
MAP(UR_DEVICE_INFO_LOCAL_MEM_SIZE, uint64_t)
MAP(UR_DEVICE_INFO_ERROR_CORRECTION_SUPPORT, ur_bool_t)
MAP(UR_DEVICE_INFO_HOST_UNIFIED_MEMORY, ur_bool_t)
MAP(UR_DEVICE_INFO_PROFILING_TIMER_RESOLUTION, size_t)
MAP(UR_DEVICE_INFO_ENDIAN_LITTLE, ur_bool_t)
MAP(UR_DEVICE_INFO_AVAILABLE, ur_bool_t)
MAP(UR_DEVICE_INFO_COMPILER_AVAILABLE, ur_bool_t)
MAP(UR_DEVICE_INFO_LINKER_AVAILABLE, ur_bool_t)
MAP(UR_DEVICE_INFO_EXECUTION_CAPABILITIES, ur_device_exec_capability_flags_t)
MAP(UR_DEVICE_INFO_QUEUE_ON_DEVICE_PROPERTIES, ur_queue_flags_t)
MAP(UR_DEVICE_INFO_QUEUE_ON_HOST_PROPERTIES, ur_queue_flags_t)
MAP(UR_DEVICE_INFO_BUILT_IN_KERNELS, std::string)
MAP(UR_DEVICE_INFO_PLATFORM, ur_platform_handle_t)
MAP(UR_DEVICE_INFO_REFERENCE_COUNT, uint32_t)
MAP(UR_DEVICE_INFO_IL_VERSION, std::string)
MAP(UR_DEVICE_INFO_NAME, std::string)
MAP(UR_DEVICE_INFO_VENDOR, std::string)
MAP(UR_DEVICE_INFO_DRIVER_VERSION, std::string)
MAP(UR_DEVICE_INFO_PROFILE, std::string)
MAP(UR_DEVICE_INFO_VERSION, std::string)
MAP(UR_DEVICE_INFO_BACKEND_RUNTIME_VERSION, std::string)
MAP(UR_DEVICE_INFO_EXTENSIONS, std::string)
MAP(UR_DEVICE_INFO_PRINTF_BUFFER_SIZE, size_t)
MAP(UR_DEVICE_INFO_PREFERRED_INTEROP_USER_SYNC, ur_bool_t)
MAP(UR_DEVICE_INFO_PARENT_DEVICE, ur_device_handle_t)
MAP(UR_DEVICE_INFO_SUPPORTED_PARTITIONS, std::vector<ur_device_partition_t>)
MAP(UR_DEVICE_INFO_PARTITION_MAX_SUB_DEVICES, uint32_t)
MAP(UR_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN, ur_device_affinity_domain_flags_t)
MAP(UR_DEVICE_INFO_PARTITION_TYPE, std::vector<ur_device_partition_property_t>)
MAP(UR_DEVICE_INFO_MAX_NUM_SUB_GROUPS, uint32_t)
MAP(UR_DEVICE_INFO_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS, ur_bool_t)
MAP(UR_DEVICE_INFO_SUB_GROUP_SIZES_INTEL, std::vector<uint32_t>)
MAP(UR_DEVICE_INFO_USM_HOST_SUPPORT, ur_device_usm_access_capability_flags_t)
MAP(UR_DEVICE_INFO_USM_DEVICE_SUPPORT, ur_device_usm_access_capability_flags_t)
MAP(UR_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT, ur_device_usm_access_capability_flags_t)
MAP(UR_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT, ur_device_usm_access_capability_flags_t)
MAP(UR_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT, ur_device_usm_access_capability_flags_t)
// Manually changed std::vector<uint8_t> -> std::array<uint8_t, 16>
MAP(UR_DEVICE_INFO_UUID, std::array<uint8_t, 16>)
MAP(UR_DEVICE_INFO_PCI_ADDRESS, std::string)
MAP(UR_DEVICE_INFO_GPU_EU_COUNT, uint32_t)
MAP(UR_DEVICE_INFO_GPU_EU_SIMD_WIDTH, uint32_t)
MAP(UR_DEVICE_INFO_GPU_EU_SLICES, uint32_t)
MAP(UR_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE, uint32_t)
MAP(UR_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE, uint32_t)
MAP(UR_DEVICE_INFO_GPU_HW_THREADS_PER_EU, uint32_t)
MAP(UR_DEVICE_INFO_MAX_MEMORY_BANDWIDTH, uint64_t)
MAP(UR_DEVICE_INFO_IMAGE_SRGB, ur_bool_t)
MAP(UR_DEVICE_INFO_BUILD_ON_SUBDEVICE, ur_bool_t)
MAP(UR_DEVICE_INFO_ATOMIC_64, ur_bool_t)
MAP(UR_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES, ur_memory_order_capability_flags_t)
MAP(UR_DEVICE_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES, ur_memory_scope_capability_flags_t)
MAP(UR_DEVICE_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES, ur_memory_order_capability_flags_t)
MAP(UR_DEVICE_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES, ur_memory_scope_capability_flags_t)
// Deprecated, we're not using it, so comment out to avoid warnings:
// MAP(UR_DEVICE_INFO_BFLOAT16, ur_bool_t)
MAP(UR_DEVICE_INFO_MAX_COMPUTE_QUEUE_INDICES, uint32_t)
MAP(UR_DEVICE_INFO_KERNEL_SET_SPECIALIZATION_CONSTANTS, ur_bool_t)
MAP(UR_DEVICE_INFO_MEMORY_BUS_WIDTH, uint32_t)
MAP(UR_DEVICE_INFO_MAX_WORK_GROUPS_3D, std::array<size_t, 3>)
MAP(UR_DEVICE_INFO_ASYNC_BARRIER, ur_bool_t)
MAP(UR_DEVICE_INFO_MEM_CHANNEL_SUPPORT, ur_bool_t)
MAP(UR_DEVICE_INFO_HOST_PIPE_READ_WRITE_SUPPORT, ur_bool_t)
MAP(UR_DEVICE_INFO_MAX_REGISTERS_PER_WORK_GROUP, uint32_t)
MAP(UR_DEVICE_INFO_IP_VERSION, uint32_t)
MAP(UR_DEVICE_INFO_VIRTUAL_MEMORY_SUPPORT, ur_bool_t)
MAP(UR_DEVICE_INFO_ESIMD_SUPPORT, ur_bool_t)
MAP(UR_DEVICE_INFO_COMPONENT_DEVICES, std::vector<ur_device_handle_t>)
MAP(UR_DEVICE_INFO_COMPOSITE_DEVICE, ur_device_handle_t)
MAP(UR_DEVICE_INFO_GLOBAL_VARIABLE_SUPPORT, ur_bool_t)
MAP(UR_DEVICE_INFO_USM_POOL_SUPPORT, ur_bool_t)
MAP(UR_DEVICE_INFO_NUM_COMPUTE_UNITS, uint32_t)
MAP(UR_DEVICE_INFO_PROGRAM_SET_SPECIALIZATION_CONSTANTS, ur_bool_t)
MAP(UR_DEVICE_INFO_USE_NATIVE_ASSERT, ur_bool_t)
MAP(UR_DEVICE_INFO_CURRENT_CLOCK_THROTTLE_REASONS, ur_device_throttle_reasons_flags_t)
MAP(UR_DEVICE_INFO_FAN_SPEED, int32_t)
MAP(UR_DEVICE_INFO_MIN_POWER_LIMIT, int32_t)
MAP(UR_DEVICE_INFO_MAX_POWER_LIMIT, int32_t)
MAP(UR_DEVICE_INFO_BFLOAT16_CONVERSIONS_NATIVE, ur_bool_t)
MAP(UR_DEVICE_INFO_KERNEL_LAUNCH_CAPABILITIES, ur_kernel_launch_properties_flags_t)
// Manually changed std::vector<uint8_t> -> std::array<uint8_t, 8>
MAP(UR_DEVICE_INFO_LUID, std::array<uint8_t, 8>)
MAP(UR_DEVICE_INFO_NODE_MASK, uint32_t)

// These aren't present in the specification, extracted from ur_api.h
// instead.
MAP(UR_DEVICE_INFO_2D_BLOCK_ARRAY_CAPABILITIES_EXP, ur_exp_device_2d_block_array_capability_flags_t)
MAP(UR_DEVICE_INFO_ASYNC_USM_ALLOCATIONS_SUPPORT_EXP, ur_bool_t)
MAP(UR_DEVICE_INFO_BINDLESS_IMAGES_1D_USM_SUPPORT_EXP, ur_bool_t)
MAP(UR_DEVICE_INFO_BINDLESS_IMAGES_2D_USM_SUPPORT_EXP, ur_bool_t)
MAP(UR_DEVICE_INFO_BINDLESS_IMAGES_GATHER_SUPPORT_EXP, ur_bool_t)
MAP(UR_DEVICE_INFO_BINDLESS_IMAGES_SHARED_USM_SUPPORT_EXP, ur_bool_t)
MAP(UR_DEVICE_INFO_BINDLESS_IMAGES_SUPPORT_EXP, ur_bool_t)
MAP(UR_DEVICE_INFO_BINDLESS_SAMPLED_IMAGE_FETCH_1D_SUPPORT_EXP, ur_bool_t)
MAP(UR_DEVICE_INFO_BINDLESS_SAMPLED_IMAGE_FETCH_1D_USM_SUPPORT_EXP, ur_bool_t)
MAP(UR_DEVICE_INFO_BINDLESS_SAMPLED_IMAGE_FETCH_2D_SUPPORT_EXP, ur_bool_t)
MAP(UR_DEVICE_INFO_BINDLESS_SAMPLED_IMAGE_FETCH_2D_USM_SUPPORT_EXP, ur_bool_t)
MAP(UR_DEVICE_INFO_BINDLESS_SAMPLED_IMAGE_FETCH_3D_SUPPORT_EXP, ur_bool_t)
MAP(UR_DEVICE_INFO_BINDLESS_SAMPLE_1D_USM_SUPPORT_EXP, ur_bool_t)
MAP(UR_DEVICE_INFO_BINDLESS_SAMPLE_2D_USM_SUPPORT_EXP, ur_bool_t)
MAP(UR_DEVICE_INFO_BINDLESS_UNIQUE_ADDRESSING_PER_DIM_SUPPORT_EXP, ur_bool_t)
MAP(UR_DEVICE_INFO_CUBEMAP_SEAMLESS_FILTERING_SUPPORT_EXP, ur_bool_t)
MAP(UR_DEVICE_INFO_CUBEMAP_SUPPORT_EXP, ur_bool_t)
MAP(UR_DEVICE_INFO_EXTERNAL_MEMORY_IMPORT_SUPPORT_EXP, ur_bool_t)
MAP(UR_DEVICE_INFO_EXTERNAL_SEMAPHORE_IMPORT_SUPPORT_EXP, ur_bool_t)
MAP(UR_DEVICE_INFO_IMAGE_ARRAY_SUPPORT_EXP, ur_bool_t)
MAP(UR_DEVICE_INFO_IMAGE_PITCH_ALIGN_EXP, uint32_t)
MAP(UR_DEVICE_INFO_MAX_IMAGE_LINEAR_HEIGHT_EXP, size_t)
MAP(UR_DEVICE_INFO_MAX_IMAGE_LINEAR_PITCH_EXP, size_t)
MAP(UR_DEVICE_INFO_MAX_IMAGE_LINEAR_WIDTH_EXP, size_t)
MAP(UR_DEVICE_INFO_MIPMAP_ANISOTROPY_SUPPORT_EXP, ur_bool_t)
MAP(UR_DEVICE_INFO_MIPMAP_LEVEL_REFERENCE_SUPPORT_EXP, ur_bool_t)
// Previous use was `float`, ur_api_t has `uint32_t` in the docstring.
MAP(UR_DEVICE_INFO_MIPMAP_MAX_ANISOTROPY_EXP, uint32_t)
MAP(UR_DEVICE_INFO_MIPMAP_SUPPORT_EXP, ur_bool_t)
MAP(UR_DEVICE_INFO_TIMESTAMP_RECORDING_SUPPORT_EXP, ur_bool_t)
MAP(UR_DEVICE_INFO_KERNEL_LAUNCH_CAPABILITIES, ur_kernel_launch_properties_flags_t)
MAP(UR_DEVICE_INFO_MEMORY_EXPORT_EXPORTABLE_DEVICE_MEM_EXP, ur_bool_t)
// clang-format on
