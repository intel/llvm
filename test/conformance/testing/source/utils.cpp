#include <string_view>
#include <uur/assert.h>
#include <uur/utils.h>

namespace uur {

std::vector<std::string> split(const std::string &str, char delim) {
    const auto n_words = std::count(str.begin(), str.end(), delim) + 1;
    std::vector<std::string> items;
    items.reserve(n_words);

    size_t start;
    size_t end = 0;
    while ((start = str.find_first_not_of(delim, end)) != std::string::npos) {
        end = str.find(delim, start);
        items.push_back(str.substr(start, end - start));
    }
    return items;
}

ur_device_type_t GetDeviceType(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<ur_device_type_t>(device, UR_DEVICE_INFO_TYPE);
    UUR_ASSERT(info.has_value(), "Failed to get device type.");
    return info.value();
}

uint32_t GetDeviceVendorId(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<uint32_t>(device, UR_DEVICE_INFO_VENDOR_ID);
    UUR_ASSERT(info.has_value(), "Failed to get device vendor ID.");
    return info.value();
}

uint32_t GetDeviceId(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<uint32_t>(device, UR_DEVICE_INFO_DEVICE_ID);
    UUR_ASSERT(info.has_value(), "Failed to get device ID.");
    return info.value();
}

uint32_t GetDeviceMaxComputeUnits(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<uint32_t>(device, UR_DEVICE_INFO_MAX_COMPUTE_UNITS);
    UUR_ASSERT(info.has_value(), "Failed to get device max compute units.");
    return info.value();
}

uint32_t GetDeviceMaxWorkItemDimensions(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<uint32_t>(
        device, UR_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS);
    UUR_ASSERT(info.has_value(), "Failed to get device max work dimensions.");
    return info.value();
}

std::vector<size_t> GetDeviceMaxWorkItemSizes(ur_device_handle_t device) {
    size_t size = 0;
    ur_result_t result = urDeviceGetInfo(
        device, UR_DEVICE_INFO_MAX_WORK_ITEM_SIZES, 0, nullptr, &size);
    if (result != UR_RESULT_SUCCESS || size == 0) {
        UUR_ABORT("urDeviceGetInfo failed: %d.", result);
    }
    size_t work_item_sizes_length = size / sizeof(size_t);
    std::vector<size_t> work_item_sizes(work_item_sizes_length);
    result = urDeviceGetInfo(device, UR_DEVICE_INFO_MAX_WORK_ITEM_SIZES, size,
                             work_item_sizes.data(), nullptr);
    if (result != UR_RESULT_SUCCESS) {
        UUR_ABORT("urDeviceGetInfo failed: %d.", result);
    }
    return work_item_sizes;
}

size_t GetDeviceMaxWorkGroupSize(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<size_t>(device, UR_DEVICE_INFO_MAX_WORK_GROUP_SIZE);
    UUR_ASSERT(info.has_value(), "Failed to get device max work group size.");
    return info.value();
}

ur_fp_capability_flags_t
GetDeviceSingleFPCapabilities(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<ur_fp_capability_flags_t>(
        device, UR_DEVICE_INFO_SINGLE_FP_CONFIG);
    UUR_ASSERT(info.has_value(),
               "Failed to get device single FP capabilities.");
    return info.value();
}

ur_fp_capability_flags_t
GetDeviceHalfFPCapabilities(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<ur_fp_capability_flags_t>(
        device, UR_DEVICE_INFO_HALF_FP_CONFIG);
    UUR_ASSERT(info.has_value(), "Failed to get device half FP capabilities.");
    return info.value();
}

ur_fp_capability_flags_t
GetDeviceDoubleFPCapabilities(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<ur_fp_capability_flags_t>(
        device, UR_DEVICE_INFO_DOUBLE_FP_CONFIG);
    UUR_ASSERT(info.has_value(),
               "Failed to get device double FP capabilities.");
    return info.value();
}

ur_queue_flags_t GetDeviceQueueProperties(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<ur_queue_flags_t>(
        device, UR_DEVICE_INFO_QUEUE_PROPERTIES);
    UUR_ASSERT(info.has_value(), "Failed to get device queue properties.");
    return info.value();
}

uint32_t GetDevicePreferredVectorWidthChar(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<uint32_t>(
        device, UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_CHAR);
    UUR_ASSERT(info.has_value(),
               "Failed to get device preferred vector width char.");
    return info.value();
}

uint32_t GetDevicePreferredVectorWidthShort(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<uint32_t>(
        device, UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_SHORT);
    UUR_ASSERT(info.has_value(),
               "Failed to get device preferred vector width short.");
    return info.value();
}

uint32_t GetDevicePreferredVectorWidthInt(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<uint32_t>(
        device, UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_INT);
    UUR_ASSERT(info.has_value(),
               "Failed to get device preferred vector width int.");
    return info.value();
}

uint32_t GetDevicePreferredVectorWidthLong(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<uint32_t>(
        device, UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_LONG);
    UUR_ASSERT(info.has_value(),
               "Failed to get device preferred vector width long.");
    return info.value();
}

uint32_t GetDevicePreferredVectorWidthFloat(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<uint32_t>(
        device, UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_FLOAT);
    UUR_ASSERT(info.has_value(),
               "Failed to get device preferred vector width float.");
    return info.value();
}

uint32_t GetDevicePreferredVectorWidthDouble(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<uint32_t>(
        device, UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_DOUBLE);
    UUR_ASSERT(info.has_value(),
               "Failed to get device preferred vector width double.");
    return info.value();
}

uint32_t GetDevicePreferredVectorWidthHalf(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<uint32_t>(
        device, UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_HALF);
    UUR_ASSERT(info.has_value(),
               "Failed to get device preferred vector width half.");
    return info.value();
}

uint32_t GetDeviceNativeVectorWithChar(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<uint32_t>(
        device, UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR);
    UUR_ASSERT(info.has_value(),
               "Failed to get device native vector width char.");
    return info.value();
}

uint32_t GetDeviceNativeVectorWithShort(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<uint32_t>(
        device, UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT);
    UUR_ASSERT(info.has_value(),
               "Failed to get device native vector width short.");
    return info.value();
}

uint32_t GetDeviceNativeVectorWithInt(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<uint32_t>(device, UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT);
    UUR_ASSERT(info.has_value(),
               "Failed to get device native vector width int.");
    return info.value();
}

uint32_t GetDeviceNativeVectorWithLong(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<uint32_t>(
        device, UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG);
    UUR_ASSERT(info.has_value(),
               "Failed to get device native vector width long.");
    return info.value();
}

uint32_t GetDeviceNativeVectorWithFloat(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<uint32_t>(
        device, UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT);
    UUR_ASSERT(info.has_value(),
               "Failed to get device native vector width float.");
    return info.value();
}

uint32_t GetDeviceNativeVectorWithDouble(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<uint32_t>(
        device, UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE);
    UUR_ASSERT(info.has_value(),
               "Failed to get device native vector width double.");
    return info.value();
}

uint32_t GetDeviceNativeVectorWithHalf(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<uint32_t>(
        device, UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF);
    UUR_ASSERT(info.has_value(),
               "Failed to get device native vector width half.");
    return info.value();
}

uint32_t GetDeviceMaxClockFrequency(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<uint32_t>(device, UR_DEVICE_INFO_MAX_CLOCK_FREQUENCY);
    UUR_ASSERT(info.has_value(), "Failed to get device max clock frequency.");
    return info.value();
}

uint32_t GetDeviceMemoryClockRate(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<uint32_t>(device, UR_DEVICE_INFO_MEMORY_CLOCK_RATE);
    UUR_ASSERT(info.has_value(), "Failed to get device memory clock rate.");
    return info.value();
}

uint32_t GetDeviceAddressBits(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<uint32_t>(device, UR_DEVICE_INFO_ADDRESS_BITS);
    UUR_ASSERT(info.has_value(), "Failed to get device address bits.");
    return info.value();
}

uint64_t GetDeviceMaxMemAllocSize(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<uint64_t>(device, UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE);
    UUR_ASSERT(info.has_value(), "Failed to get device max mem alloc size.");
    return info.value();
}

bool GetDeviceImageSupport(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<bool>(device, UR_DEVICE_INFO_IMAGE_SUPPORTED);
    UUR_ASSERT(info.has_value(), "Failed to get device immage support.");
    return info.value();
}

uint32_t GetDeviceMaxReadImageArgs(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<uint32_t>(device, UR_DEVICE_INFO_MAX_READ_IMAGE_ARGS);
    UUR_ASSERT(info.has_value(), "Failed to get device max read image args.");
    return info.value();
}

uint32_t GetDeviceMaxWriteImageArgs(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<uint32_t>(device, UR_DEVICE_INFO_MAX_WRITE_IMAGE_ARGS);
    UUR_ASSERT(info.has_value(), "Failed to get device max write image args.");
    return info.value();
}

uint32_t GetDeviceMaxReadWriteImageArgs(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<uint32_t>(
        device, UR_DEVICE_INFO_MAX_READ_WRITE_IMAGE_ARGS);
    UUR_ASSERT(info.has_value(),
               "Failed to get device max read/write image args.");
    return info.value();
}

size_t GetDeviceImage2DMaxWidth(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<size_t>(device, UR_DEVICE_INFO_IMAGE2D_MAX_WIDTH);
    UUR_ASSERT(info.has_value(), "Failed to get device image2d max width.");
    return info.value();
}

size_t GetDeviceImage2DMaxHeight(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<size_t>(device, UR_DEVICE_INFO_IMAGE2D_MAX_HEIGHT);
    UUR_ASSERT(info.has_value(), "Failed to get device image2d max height.");
    return info.value();
}

size_t GetDeviceImage3DMaxWidth(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<size_t>(device, UR_DEVICE_INFO_IMAGE3D_MAX_WIDTH);
    UUR_ASSERT(info.has_value(), "Failed to get device image3d max width.");
    return info.value();
}

size_t GetDeviceImage3DMaxHeight(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<size_t>(device, UR_DEVICE_INFO_IMAGE3D_MAX_HEIGHT);
    UUR_ASSERT(info.has_value(), "Failed to get device image3d max height.");
    return info.value();
}

size_t GetDeviceImage3DMaxDepth(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<size_t>(device, UR_DEVICE_INFO_IMAGE3D_MAX_DEPTH);
    UUR_ASSERT(info.has_value(), "Failed to get device image3d max depth.");
    return info.value();
}

size_t GetDeviceImageMaxBufferSize(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<size_t>(device, UR_DEVICE_INFO_IMAGE_MAX_BUFFER_SIZE);
    UUR_ASSERT(info.has_value(), "Failed to get device max buffer size.");
    return info.value();
}

size_t GetDeviceImageMaxArraySize(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<size_t>(device, UR_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE);
    UUR_ASSERT(info.has_value(), "Failed to get device max array size.");
    return info.value();
}

uint32_t GetDeviceMaxSamplers(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<uint32_t>(device, UR_DEVICE_INFO_MAX_SAMPLERS);
    UUR_ASSERT(info.has_value(), "Failed to get device max samplers.");
    return info.value();
}

size_t GetDeviceMaxParameterSize(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<size_t>(device, UR_DEVICE_INFO_MAX_PARAMETER_SIZE);
    UUR_ASSERT(info.has_value(), "Failed to get device max parameter size.");
    return info.value();
}

uint32_t GetDeviceMemBaseAddressAlign(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<uint32_t>(device, UR_DEVICE_INFO_MEM_BASE_ADDR_ALIGN);
    UUR_ASSERT(info.has_value(),
               "Failed to get device mem base address align.");
    return info.value();
}

ur_device_mem_cache_type_t GetDeviceMemCacheType(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<ur_device_mem_cache_type_t>(
        device, UR_DEVICE_INFO_GLOBAL_MEM_CACHE_TYPE);
    UUR_ASSERT(info.has_value(), "Failed to get device global mem cache type.");
    return info.value();
}

uint32_t GetDeviceMemCachelineSize(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<uint32_t>(
        device, UR_DEVICE_INFO_GLOBAL_MEM_CACHELINE_SIZE);
    UUR_ASSERT(info.has_value(), "Failed to get device mem cache line size.");
    return info.value();
}

uint64_t GetDeviceMemCacheSize(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<uint64_t>(device, UR_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE);
    UUR_ASSERT(info.has_value(), "Failed to get device mem cache size.");
    return info.value();
}

uint64_t GetDeviceGlobalMemSize(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<uint64_t>(device, UR_DEVICE_INFO_GLOBAL_MEM_SIZE);
    UUR_ASSERT(info.has_value(), "Failed to get device global mem size.");
    return info.value();
}

uint64_t GetDeviceGlobalMemFree(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<uint64_t>(device, UR_DEVICE_INFO_GLOBAL_MEM_FREE);
    UUR_ASSERT(info.has_value(), "Failed to get device global mem size.");
    return info.value();
}

uint64_t GetDeviceMaxConstantBufferSize(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<uint64_t>(
        device, UR_DEVICE_INFO_MAX_CONSTANT_BUFFER_SIZE);
    UUR_ASSERT(info.has_value(),
               "Failed to get device max constant buffer size.");
    return info.value();
}

uint32_t GetDeviceMaxConstantArgs(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<uint32_t>(device, UR_DEVICE_INFO_MAX_CONSTANT_ARGS);
    UUR_ASSERT(info.has_value(), "Failed to get device max constant args.");
    return info.value();
}

ur_device_local_mem_type_t GetDeviceLocalMemType(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<ur_device_local_mem_type_t>(
        device, UR_DEVICE_INFO_LOCAL_MEM_TYPE);
    UUR_ASSERT(info.has_value(), "Failed to get device local mem type.");
    return info.value();
}

uint64_t GetDeviceLocalMemSize(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<uint64_t>(device, UR_DEVICE_INFO_LOCAL_MEM_SIZE);
    UUR_ASSERT(info.has_value(), "Failed to get device local mem size.");
    return info.value();
}

bool GetDeviceErrorCorrectionSupport(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<bool>(device, UR_DEVICE_INFO_ERROR_CORRECTION_SUPPORT);
    UUR_ASSERT(info.has_value(),
               "Failed to get device error correction support.");
    return info.value();
}

// host usm

size_t GetDeviceProfilingTimerResolution(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<size_t>(
        device, UR_DEVICE_INFO_PROFILING_TIMER_RESOLUTION);
    UUR_ASSERT(info.has_value(),
               "Failed to get device profiling timer resolution.");
    return info.value();
}

bool GetDeviceLittleEndian(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<bool>(device, UR_DEVICE_INFO_ENDIAN_LITTLE);
    UUR_ASSERT(info.has_value(), "Failed to get device is little endian.");
    return info.value();
}

bool GetDeviceAvailable(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<bool>(device, UR_DEVICE_INFO_AVAILABLE);
    UUR_ASSERT(info.has_value(), "Failed to get device is available.");
    return info.value();
}

bool GetDeviceCompilerAvailable(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<bool>(device, UR_DEVICE_INFO_COMPILER_AVAILABLE);
    UUR_ASSERT(info.has_value(), "Failed to get device compiler available.");
    return info.value();
}

bool GetDeviceLinkerAvailable(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<bool>(device, UR_DEVICE_INFO_LINKER_AVAILABLE);
    UUR_ASSERT(info.has_value(), "Failed to get device linker available.");
    return info.value();
}

ur_device_exec_capability_flags_t
GetDeviceExecutionCapabilities(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<ur_device_exec_capability_flags_t>(
        device, UR_DEVICE_INFO_EXECUTION_CAPABILITIES);
    UUR_ASSERT(info.has_value(),
               "Failed to get device execution capabilities.");
    return info.value();
}

ur_queue_flags_t GetDeviceQueueOnDeviceProperties(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<ur_queue_flags_t>(
        device, UR_DEVICE_INFO_QUEUE_ON_DEVICE_PROPERTIES);
    UUR_ASSERT(info.has_value(), "Failed to get device on device properties.");
    return info.value();
}

ur_queue_flags_t GetDeviceQueueOnHostProperties(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<ur_queue_flags_t>(
        device, UR_DEVICE_INFO_QUEUE_ON_HOST_PROPERTIES);
    UUR_ASSERT(info.has_value(), "Failed to get device on host properties.");
    return info.value();
}

std::vector<std::string> GetDeviceBuiltInKernels(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<std::string>(device, UR_DEVICE_INFO_BUILT_IN_KERNELS);
    UUR_ASSERT(info.has_value(), "Failed to get device builtin kernels.");
    const auto kernels_str = info.value();
    return split(kernels_str, ';');
}

ur_platform_handle_t GetDevicePlatform(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<ur_platform_handle_t>(device, UR_DEVICE_INFO_PLATFORM);
    UUR_ASSERT(info.has_value(), "Failed to get device platform.");
    return info.value();
}

uint32_t GetDeviceReferenceCount(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<uint32_t>(device, UR_DEVICE_INFO_REFERENCE_COUNT);
    UUR_ASSERT(info.has_value(), "Failed to get device reference count.");
    return info.value();
}

std::string GetDeviceILVersion(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<std::string>(device, UR_DEVICE_INFO_IL_VERSION);
    UUR_ASSERT(info.has_value(), "Failed to get device IL version.");
    return info.value();
}

std::string
GetDeviceName_(ur_device_handle_t device); // TODO - this is a duplicate

std::string GetDeviceVendor(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<std::string>(device, UR_DEVICE_INFO_NAME);
    UUR_ASSERT(info.has_value(), "Failed to get device name.");
    return info.value();
}

std::string GetDeviceDriverVersion(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<std::string>(device, UR_DEVICE_INFO_VENDOR);
    UUR_ASSERT(info.has_value(), "Failed to get device vendor.");
    return info.value();
}

std::string GetDeviceProfile(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<std::string>(device, UR_DEVICE_INFO_DRIVER_VERSION);
    UUR_ASSERT(info.has_value(), "Failed to get device version.");
    return info.value();
}

std::string GetDeviceVersion(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<std::string>(device, UR_DEVICE_INFO_PROFILE);
    UUR_ASSERT(info.has_value(), "Failed to get device profile.");
    return info.value();
}

std::string GetDeviceBackendRuntimeVersion(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<std::string>(device, UR_DEVICE_INFO_VERSION);
    UUR_ASSERT(info.has_value(), "Failed to get device version.");
    return info.value();
}

std::vector<std::string> GetDeviceExtensions(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<std::string>(
        device, UR_DEVICE_INFO_BACKEND_RUNTIME_VERSION);
    UUR_ASSERT(info.has_value(), "Failed to get device backend runtime.");
    return split(info.value(), ' ');
}

size_t GetDevicePrintfBufferSize(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<size_t>(device, UR_DEVICE_INFO_PRINTF_BUFFER_SIZE);
    UUR_ASSERT(info.has_value(), "Failed to get device printf buffer size.");
    return info.value();
}

bool GetDevicePreferredInteropUserSync(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<bool>(device, UR_DEVICE_INFO_PREFERRED_INTEROP_USER_SYNC);
    UUR_ASSERT(info.has_value(),
               "Failed to get device preferred interop user sync.");
    return info.value();
}

ur_device_handle_t GetDeviceParentDevice(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<ur_device_handle_t>(device, UR_DEVICE_INFO_PARENT_DEVICE);
    UUR_ASSERT(info.has_value(), "Failed to get device parent device.");
    return info.value();
}

std::vector<ur_device_partition_property_t>
GetDevicePartitionProperties(ur_device_handle_t device) {
    size_t size = 0;
    ur_result_t result = urDeviceGetInfo(
        device, UR_DEVICE_INFO_PARTITION_PROPERTIES, 0, nullptr, &size);
    if (result != UR_RESULT_SUCCESS || size == 0) {
        UUR_ABORT("urDeviceGetInfo failed: %d.", result);
    }
    std::vector<ur_device_partition_property_t> properties(
        size / sizeof(ur_device_partition_property_t));
    result = urDeviceGetInfo(device, UR_DEVICE_INFO_PARTITION_PROPERTIES, size,
                             properties.data(), &size);
    if (result != UR_RESULT_SUCCESS) {
        UUR_ABORT("urDeviceGetInfo failed: %d.", result);
    }
    return properties;
}

uint32_t GetDevicePartitionMaxSubDevices(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<uint32_t>(
        device, UR_DEVICE_INFO_PARTITION_MAX_SUB_DEVICES);
    UUR_ASSERT(info.has_value(),
               "Failed to get device partition max sub devices.");
    return info.value();
}

ur_device_affinity_domain_flags_t
GetDevicePartitionAffinityDomainFlags(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<ur_device_affinity_domain_flags_t>(
        device, UR_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN);
    UUR_ASSERT(info.has_value(),
               "Failed to get device partition affinity domain.");
    return info.value();
}

std::vector<ur_device_partition_property_t>
GetDevicePartitionType(ur_device_handle_t device) {
    size_t size = 0;
    ur_result_t result = urDeviceGetInfo(device, UR_DEVICE_INFO_PARTITION_TYPE,
                                         0, nullptr, &size);
    if (result != UR_RESULT_SUCCESS || size == 0) {
        UUR_ABORT("urDeviceGetInfo failed: %d.", result);
    }
    size_t elem_count = size / sizeof(ur_device_partition_property_t);
    std::vector<ur_device_partition_property_t> props(elem_count);
    result = urDeviceGetInfo(device, UR_DEVICE_INFO_PARTITION_TYPE, size,
                             props.data(), nullptr);
    if (result != UR_RESULT_SUCCESS) {
        UUR_ABORT("urDeviceGetInfo failed: %d.", result);
    }
    return props;
}

uint32_t GetDeviceMaxNumberSubGroups(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<uint32_t>(device, UR_DEVICE_INFO_MAX_NUM_SUB_GROUPS);
    UUR_ASSERT(info.has_value(), "Failed to get device max sub groups.");
    return info.value();
}

bool GetDeviceSubGroupIndependentForwardProgress(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<bool>(
        device, UR_DEVICE_INFO_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS);
    UUR_ASSERT(info.has_value(),
               "Failed to get device independent forward progress.");
    return info.value();
}

std::vector<uint32_t> GetDeviceSubGroupSizesIntel(ur_device_handle_t device) {
    size_t size = 0;
    ur_result_t result = urDeviceGetInfo(
        device, UR_DEVICE_INFO_SUB_GROUP_SIZES_INTEL, 0, nullptr, &size);
    if (result != UR_RESULT_SUCCESS || size == 0) {
        UUR_ABORT("urDeviceGetInfo failed: %d.", result);
    }
    size_t elem_count = size / sizeof(uint32_t);
    std::vector<uint32_t> sub_group_sizes(elem_count);
    result = urDeviceGetInfo(device, UR_DEVICE_INFO_SUB_GROUP_SIZES_INTEL, size,
                             sub_group_sizes.data(), nullptr);
    if (result != UR_RESULT_SUCCESS) {
        UUR_ABORT("urDeviceGetInfo failed: %d.", result);
    }
    return sub_group_sizes;
}

bool GetDeviceUSMHostSupport(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<bool>(device, UR_DEVICE_INFO_USM_HOST_SUPPORT);
    UUR_ASSERT(info.has_value(), "Failed to get device host usm support.");
    return info.value();
}

bool GetDeviceUSMDeviceSupport(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<bool>(device, UR_DEVICE_INFO_USM_DEVICE_SUPPORT);
    UUR_ASSERT(info.has_value(), "Failed to get device device usm support.");
    return info.value();
}

bool GetDeviceUSMSingleSharedSupport(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<bool>(device, UR_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT);
    UUR_ASSERT(info.has_value(),
               "Failed to get device single shared usm support.");
    return info.value();
}

bool GetDeviceUSMCrossSharedSupport(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<bool>(device, UR_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT);
    UUR_ASSERT(info.has_value(),
               "Failed to get device cross shared usm support.");
    return info.value();
}

bool GetDeviceUSMSystemSharedSupport(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<bool>(device, UR_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT);
    UUR_ASSERT(info.has_value(),
               "Failed to get device system shared usm support.");
    return info.value();
}

std::string GetDeviceUUID(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<std::string>(device, UR_DEVICE_INFO_UUID);
    UUR_ASSERT(info.has_value(), "Failed to get device UUID.");
    return info.value();
}

std::string GetDevicePCIAddress(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<std::string>(device, UR_DEVICE_INFO_PCI_ADDRESS);
    UUR_ASSERT(info.has_value(), "Failed to get device PCI address.");
    return info.value();
}

uint32_t GetDeviceGPUEUCount(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<bool>(device, UR_DEVICE_INFO_GPU_EU_COUNT);
    UUR_ASSERT(info.has_value(), "Failed to get device GPU EU count.");
    return info.value();
}

uint32_t GetDeviceGPUEUSIMDWidth(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<uint32_t>(device, UR_DEVICE_INFO_GPU_EU_SIMD_WIDTH);
    UUR_ASSERT(info.has_value(), "Failed to get device GPU EU simd width.");
    return info.value();
}

uint32_t GetDeviceGPUEUSlices(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<uint32_t>(device, UR_DEVICE_INFO_GPU_EU_SLICES);
    UUR_ASSERT(info.has_value(), "Failed to get device GPU EU slices.");
    return info.value();
}

uint32_t GetDeviceGPUSubslicesPerSlice(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<uint32_t>(device, UR_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE);
    UUR_ASSERT(info.has_value(),
               "Failed to get device GPU subslices per slice.");
    return info.value();
}

uint32_t GetDeviceMaxMemoryBandwidth(ur_device_handle_t device) {
    const auto info =
        GetDeviceInfo<uint32_t>(device, UR_DEVICE_INFO_MAX_MEMORY_BANDWIDTH);
    UUR_ASSERT(info.has_value(), "Failed to get device max memory bandwidth.");
    return info.value();
}

bool GetDeviceImageSRGB(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<bool>(device, UR_DEVICE_INFO_IMAGE_SRGB);
    UUR_ASSERT(info.has_value(), "Failed to get device image SRGB.");
    return info.value();
}

bool GetDeviceAtomic64Support(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<bool>(device, UR_DEVICE_INFO_ATOMIC_64);
    UUR_ASSERT(info.has_value(), "Failed to get device atomic 64.");
    return info.value();
}

ur_memory_order_capability_flags_t
GetDeviceMemoryOrderCapabilities(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<ur_memory_order_capability_flags_t>(
        device, UR_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES);
    UUR_ASSERT(info.has_value(),
               "Failed to get device atomic memory order capabilities.");
    return info.value();
}

ur_memory_scope_capability_flags_t
GetDeviceMemoryScopeCapabilities(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<ur_memory_scope_capability_flags_t>(
        device, UR_DEVICE_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES);
    UUR_ASSERT(info.has_value(),
               "Failed to get device atomic memory scope capabilities.");
    return info.value();
}

bool GetDeviceBFloat16Support(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<bool>(device, UR_DEVICE_INFO_BFLOAT16);
    UUR_ASSERT(info.has_value(), "Failed to get device bfloat16 support.");
    return info.value();
}

uint32_t GetDeviceMaxComputeQueueIndices(ur_device_handle_t device) {
    const auto info = GetDeviceInfo<uint32_t>(
        device, UR_DEVICE_INFO_MAX_COMPUTE_QUEUE_INDICES);
    UUR_ASSERT(info.has_value(),
               "Failed to get device max compute queue indices.");
    return info.value();
}

} // namespace uur
