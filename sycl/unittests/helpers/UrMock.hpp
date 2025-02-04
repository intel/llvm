//==------------- UrMock.hpp --- Mock unit testing library -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This mini-library provides facilities to test the DPC++ Runtime behavior upon
// specific results of the underlying low-level API calls. By exploiting the
// Adapter Interface API, the stored addresses of the actual adapter-specific
// implementations can be overwritten to point at user-defined mock functions.
//
// To make testing independent of existing adapters and devices, all adapters
// are forcefully unloaded and the mock adapter is registered as the only
// adapter.
//
// While this could be done manually for each unit-testing scenario, the library
// aims to rule out the boilerplate, providing helper APIs which can be re-used
// by all such unit tests. The test code stemming from this can be more consise,
// with little difference from non-mock classes' usage.
//
// The following unit testing scenarios are thereby simplified:
// 1) testing the DPC++ RT management of specific UR return codes;
// 2) coverage of corner-cases related to specific data outputs
//    from underlying runtimes;
// 3) testing the order of UR API calls;
// ..., etc.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/adapter.hpp>
#include <detail/global_handler.hpp>
#include <detail/platform_impl.hpp>
#include <detail/ur.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/device.hpp>
#include <sycl/device_selector.hpp>
#include <sycl/platform.hpp>
#include <sycl/queue.hpp>

#include <list>
#include <optional>

#include <ur_mock_helpers.hpp>

namespace sycl {

inline namespace _V1 {
namespace unittest {

namespace detail = sycl::detail;

namespace MockAdapter {

inline ur_result_t mock_urPlatformGet(void *pParams) {
  auto params = reinterpret_cast<ur_platform_get_params_t *>(pParams);
  if (*params->ppNumPlatforms)
    **params->ppNumPlatforms = 1;

  if (*params->pphPlatforms && *params->pNumEntries > 0)
    *params->pphPlatforms[0] = reinterpret_cast<ur_platform_handle_t>(1);

  return UR_RESULT_SUCCESS;
}

inline ur_result_t mock_urDeviceGet(void *pParams) {
  auto params = reinterpret_cast<ur_device_get_params_t *>(pParams);
  if (*params->ppNumDevices)
    **params->ppNumDevices = 1;

  if (*params->pphDevices && *params->pNumEntries > 0)
    *params->pphDevices[0] = reinterpret_cast<ur_device_handle_t>(1);

  return UR_RESULT_SUCCESS;
}

// since we're overriding DeviceGet to return a specific fake handle we'll also
// need to override the Retain/Release functions

inline ur_result_t mock_urDeviceRetain(void *) { return UR_RESULT_SUCCESS; }
inline ur_result_t mock_urDeviceRelease(void *) { return UR_RESULT_SUCCESS; }

template <ur_backend_t Backend>
inline ur_result_t mock_urAdapterGetInfo(void *pParams) {
  auto params = reinterpret_cast<ur_adapter_get_info_params_t *>(pParams);

  if (*params->ppropName == UR_ADAPTER_INFO_BACKEND) {
    constexpr auto MockPlatformBackend = Backend;
    if (*params->ppPropValue) {
      std::memcpy(*params->ppPropValue, &MockPlatformBackend,
                  sizeof(MockPlatformBackend));
    }
    if (*params->ppPropSizeRet)
      **params->ppPropSizeRet = sizeof(MockPlatformBackend);
    return UR_RESULT_SUCCESS;
  }
  return UR_RESULT_SUCCESS;
}

template <ur_backend_t Backend>
inline ur_result_t mock_urPlatformGetInfo(void *pParams) {
  auto params = reinterpret_cast<ur_platform_get_info_params_t *>(pParams);
  constexpr char MockPlatformName[] = "Mock platform";
  constexpr char MockSupportedExtensions[] =
      "cl_khr_il_program cl_khr_subgroups cl_intel_subgroups "
      "cl_intel_subgroups_short cl_intel_required_subgroup_size ";
  switch (*params->ppropName) {
  case UR_PLATFORM_INFO_NAME: {
    if (*params->ppPropValue) {
      assert(*params->ppropSize == sizeof(MockPlatformName));
      std::memcpy(*params->ppPropValue, MockPlatformName,
                  sizeof(MockPlatformName));
    }
    if (*params->ppPropSizeRet)
      **params->ppPropSizeRet = sizeof(MockPlatformName);
    return UR_RESULT_SUCCESS;
  }
  case UR_PLATFORM_INFO_EXTENSIONS: {
    if (*params->ppPropValue) {
      assert(*params->ppropSize == sizeof(MockSupportedExtensions));
      std::memcpy(*params->ppPropValue, MockSupportedExtensions,
                  sizeof(MockSupportedExtensions));
    }
    if (*params->ppPropSizeRet)
      **params->ppPropSizeRet = sizeof(MockSupportedExtensions);
    return UR_RESULT_SUCCESS;
  }
  case UR_PLATFORM_INFO_BACKEND: {
    constexpr auto MockPlatformBackend = Backend;
    if (*params->ppPropValue) {
      std::memcpy(*params->ppPropValue, &MockPlatformBackend,
                  sizeof(MockPlatformBackend));
    }
    if (*params->ppPropSizeRet)
      **params->ppPropSizeRet = sizeof(MockPlatformBackend);
    return UR_RESULT_SUCCESS;
  }
  default: {
    constexpr const char FallbackValue[] = "str";
    constexpr size_t FallbackValueSize = sizeof(FallbackValue);
    if (*params->ppPropSizeRet)
      **params->ppPropSizeRet = FallbackValueSize;

    if (*params->ppPropValue && *params->ppropSize >= FallbackValueSize)
      std::memcpy(*params->ppPropValue, FallbackValue, FallbackValueSize);

    return UR_RESULT_SUCCESS;
  }
  }
}

inline ur_result_t mock_urDeviceGetInfo(void *pParams) {
  auto params = reinterpret_cast<ur_device_get_info_params_t *>(pParams);
  constexpr char MockDeviceName[] = "Mock device";
  constexpr char MockSupportedExtensions[] =
      "cl_khr_fp64 cl_khr_fp16 cl_khr_il_program ur_exp_command_buffer";
  switch (*params->ppropName) {
  case UR_DEVICE_INFO_TYPE: {
    // Act like any device is a GPU.
    // TODO: Should we mock more device types?
    if (*params->ppPropValue)
      *static_cast<ur_device_type_t *>(*params->ppPropValue) =
          UR_DEVICE_TYPE_GPU;
    if (*params->ppPropSizeRet)
      **params->ppPropSizeRet = sizeof(UR_DEVICE_TYPE_GPU);
    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_NAME: {
    if (*params->ppPropValue) {
      assert(*params->ppropSize == sizeof(MockDeviceName));
      std::memcpy(*params->ppPropValue, MockDeviceName, sizeof(MockDeviceName));
    }
    if (*params->ppPropSizeRet)
      **params->ppPropSizeRet = sizeof(MockDeviceName);
    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_PARENT_DEVICE: {
    if (*params->ppPropValue)
      *static_cast<ur_device_handle_t *>(*params->ppPropValue) = nullptr;
    if (*params->ppPropSizeRet)
      **params->ppPropSizeRet = sizeof(ur_device_handle_t *);
    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_EXTENSIONS: {
    if (*params->ppPropValue) {
      assert(*params->ppropSize >= sizeof(MockSupportedExtensions));
      std::memcpy(*params->ppPropValue, MockSupportedExtensions,
                  sizeof(MockSupportedExtensions));
    }
    if (*params->ppPropSizeRet)
      **params->ppPropSizeRet = sizeof(MockSupportedExtensions);
    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_USM_HOST_SUPPORT:
  case UR_DEVICE_INFO_USM_DEVICE_SUPPORT:
  case UR_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT:
  case UR_DEVICE_INFO_HOST_UNIFIED_MEMORY:
  case UR_DEVICE_INFO_AVAILABLE:
  case UR_DEVICE_INFO_LINKER_AVAILABLE:
  case UR_DEVICE_INFO_COMPILER_AVAILABLE:
  case UR_DEVICE_INFO_COMMAND_BUFFER_SUPPORT_EXP: {
    if (*params->ppPropValue)
      *static_cast<ur_bool_t *>(*params->ppPropValue) = true;
    if (*params->ppPropSizeRet)
      **params->ppPropSizeRet = sizeof(true);
    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_COMMAND_BUFFER_UPDATE_CAPABILITIES_EXP: {
    if (*params->ppPropValue)
      *static_cast<ur_device_command_buffer_update_capability_flags_t *>(
          *params->ppPropValue) =
          UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_KERNEL_ARGUMENTS |
          UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_GLOBAL_WORK_SIZE |
          UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_LOCAL_WORK_SIZE |
          UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_GLOBAL_WORK_OFFSET |
          UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_KERNEL_HANDLE;
    if (*params->ppPropSizeRet)
      **params->ppPropSizeRet =
          sizeof(ur_device_command_buffer_update_capability_flags_t);
    return UR_RESULT_SUCCESS;
  }
  // This mock GPU device has no sub-devices
  case UR_DEVICE_INFO_SUPPORTED_PARTITIONS: {
    if (*params->ppPropSizeRet) {
      **params->ppPropSizeRet = 0;
    }
    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN: {
    assert(*params->ppropSize == sizeof(ur_device_affinity_domain_flags_t));
    if (*params->ppPropValue) {
      *static_cast<ur_device_affinity_domain_flags_t *>(*params->ppPropValue) =
          0;
    }
    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_QUEUE_PROPERTIES: {
    assert(*params->ppropSize == sizeof(ur_queue_flags_t));
    if (*params->ppPropValue) {
      *static_cast<ur_queue_flags_t *>(*params->ppPropValue) =
          UR_QUEUE_FLAG_PROFILING_ENABLE;
    }
    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_COMPONENT_DEVICES:
    if (*params->ppPropValue) {
      *static_cast<ur_device_handle_t *>(*params->ppPropValue) = nullptr;
    }
    if (*params->ppPropSizeRet) {
      **params->ppPropSizeRet = 0;
    }
    return UR_RESULT_SUCCESS;
  default: {
    // In the default case we fill the return value with 0's. This may not be
    // valid for all device queries, but it will mean a consistent return value
    // for the query.
    // Any tests that need special return values should either add behavior
    // the this function or use redefineAfter with a function that adds the
    // intended behavior.
    if (*params->ppPropValue && *params->ppropSize != 0)
      std::memset(*params->ppPropValue, 0, *params->ppropSize);
    // Likewise, if the device info query asks for the size of the return value
    // we tell it there is a single byte to avoid cases where the runtime tries
    // to allocate some random amount of memory for the return value.
    if (*params->ppPropSizeRet)
      **params->ppPropSizeRet = 1;
    return UR_RESULT_SUCCESS;
  }
  }
}

inline ur_result_t mock_urProgramGetInfo(void *pParams) {
  auto params = reinterpret_cast<ur_program_get_info_params_t *>(pParams);
  switch (*params->ppropName) {
  case UR_PROGRAM_INFO_NUM_DEVICES: {
    if (*params->ppPropValue)
      *static_cast<unsigned int *>(*params->ppPropValue) = 1;
    if (*params->ppPropSizeRet)
      **params->ppPropSizeRet = sizeof(size_t);
    return UR_RESULT_SUCCESS;
  }
  case UR_PROGRAM_INFO_DEVICES: {
    if (*params->ppPropValue)
      *static_cast<ur_device_handle_t *>(*params->ppPropValue) =
          reinterpret_cast<ur_device_handle_t>(0x1);
    if (*params->ppPropSizeRet)
      **params->ppPropSizeRet = sizeof(ur_device_handle_t);
    return UR_RESULT_SUCCESS;
  }
  case UR_PROGRAM_INFO_BINARY_SIZES: {
    if (*params->ppPropValue)
      *static_cast<size_t *>(*params->ppPropValue) = 1;
    if (*params->ppPropSizeRet)
      **params->ppPropSizeRet = sizeof(size_t);
    return UR_RESULT_SUCCESS;
  }
  case UR_PROGRAM_INFO_BINARIES: {
    if (*params->ppPropValue)
      **static_cast<unsigned char **>(*params->ppPropValue) = 1;
    if (*params->ppPropSizeRet)
      **params->ppPropSizeRet = sizeof(unsigned char);
    return UR_RESULT_SUCCESS;
  }
  default: {
    // TODO: Buildlog requires this but not any actual data afterwards.
    //       This should be investigated. Should this be moved to that test?
    if (*params->ppPropSizeRet)
      **params->ppPropSizeRet = sizeof(size_t);
    return UR_RESULT_SUCCESS;
  }
  }
}

inline ur_result_t mock_urContextGetInfo(void *pParams) {
  auto params = reinterpret_cast<ur_context_get_info_params_t *>(pParams);
  switch (*params->ppropName) {
  case UR_CONTEXT_INFO_NUM_DEVICES: {
    if (*params->ppPropValue)
      *static_cast<uint32_t *>(*params->ppPropValue) = 1;
    if (*params->ppPropSizeRet)
      **params->ppPropSizeRet = sizeof(uint32_t);
    return UR_RESULT_SUCCESS;
  }
  default:
    return UR_RESULT_SUCCESS;
  }
}

inline ur_result_t mock_urQueueGetInfo(void *pParams) {
  auto params = reinterpret_cast<ur_queue_get_info_params_t *>(pParams);
  switch (*params->ppropName) {
  case UR_QUEUE_INFO_DEVICE: {
    if (*params->ppPropValue)
      *static_cast<ur_device_handle_t *>(*params->ppPropValue) =
          reinterpret_cast<ur_device_handle_t>(1);
    if (*params->ppPropSizeRet)
      **params->ppPropSizeRet = sizeof(ur_device_handle_t);
    return UR_RESULT_SUCCESS;
  }
  default:
    return UR_RESULT_SUCCESS;
  }
}

inline ur_result_t mock_urKernelGetGroupInfo(void *pParams) {
  auto params = reinterpret_cast<ur_kernel_get_group_info_params_t *>(pParams);
  switch (*params->ppropName) {
  case UR_KERNEL_GROUP_INFO_WORK_GROUP_SIZE: {
    if (*params->ppPropValue) {
      auto RealVal = reinterpret_cast<size_t *>(*params->ppPropValue);
      RealVal[0] = 0;
      RealVal[1] = 0;
      RealVal[2] = 0;
    }
    if (*params->ppPropSizeRet)
      **params->ppPropSizeRet = 3 * sizeof(size_t);
    return UR_RESULT_SUCCESS;
  }
  default: {
    return UR_RESULT_SUCCESS;
  }
  }
}

inline ur_result_t mock_urEventGetInfo(void *pParams) {
  auto params = reinterpret_cast<ur_event_get_info_params_t *>(pParams);
  switch (*params->ppropName) {
  case UR_EVENT_INFO_COMMAND_EXECUTION_STATUS: {
    if (*params->ppPropValue)
      *static_cast<ur_event_status_t *>(*params->ppPropValue) =
          UR_EVENT_STATUS_SUBMITTED;
    if (*params->ppPropSizeRet)
      **params->ppPropSizeRet = sizeof(ur_event_status_t);
    return UR_RESULT_SUCCESS;
  }
  default: {
    return UR_RESULT_SUCCESS;
  }
  }
}

inline ur_result_t
mock_urKernelSuggestMaxCooperativeGroupCountExp(void *pParams) {
  auto params = reinterpret_cast<
      ur_kernel_suggest_max_cooperative_group_count_exp_params_t *>(pParams);
  **params->ppGroupCountRet = 1;
  return UR_RESULT_SUCCESS;
}

inline ur_result_t mock_urDeviceSelectBinary(void *pParams) {
  auto params = reinterpret_cast<ur_device_select_binary_params_t *>(pParams);
  **params->ppSelectedBinary = 0;
  return UR_RESULT_SUCCESS;
}

inline ur_result_t mock_urPlatformGetBackendOption(void *pParams) {
  auto params =
      reinterpret_cast<ur_platform_get_backend_option_params_t *>(pParams);
  **params->pppPlatformOption = "";
  return UR_RESULT_SUCCESS;
}

// Returns the wall-clock timestamp of host for deviceTime and hostTime
inline ur_result_t mock_urDeviceGetGlobalTimestamps(void *pParams) {
  auto params =
      reinterpret_cast<ur_device_get_global_timestamps_params_t *>(pParams);
  using namespace std::chrono;
  auto timeNanoseconds =
      duration_cast<nanoseconds>(steady_clock::now().time_since_epoch())
          .count();
  if (*params->ppDeviceTimestamp) {
    **params->ppDeviceTimestamp = timeNanoseconds;
  }
  if (*params->ppHostTimestamp) {
    **params->ppHostTimestamp = timeNanoseconds;
  }
  return UR_RESULT_SUCCESS;
}

inline ur_result_t mock_urUsmP2PPeerAccessGetInfoExp(void *pParams) {
  auto params =
      reinterpret_cast<ur_usm_p2p_peer_access_get_info_exp_params_t *>(pParams);
  if (*params->ppPropValue)
    *static_cast<int32_t *>(*params->ppPropValue) = 1;
  if (*params->ppPropSizeRet)
    **params->ppPropSizeRet = sizeof(int32_t);

  return UR_RESULT_SUCCESS;
}

inline ur_result_t mock_urVirtualMemReserve(void *pParams) {
  auto params = reinterpret_cast<ur_virtual_mem_reserve_params_t *>(pParams);
  **params->pppStart = *params->ppStart
                           ? const_cast<void *>(*params->ppStart)
                           : mock::createDummyHandle<void *>(*params->psize);
  return UR_RESULT_SUCCESS;
}

// Create dummy command buffer handle and store the provided descriptor as the
// data
inline ur_result_t mock_urCommandBufferCreateExp(void *pParams) {
  auto params =
      reinterpret_cast<ur_command_buffer_create_exp_params_t *>(pParams);
  const ur_exp_command_buffer_desc_t *descPtr = *(params->ppCommandBufferDesc);
  ur_exp_command_buffer_handle_t *retCmdBuffer = *params->pphCommandBuffer;
  *retCmdBuffer = mock::createDummyHandle<ur_exp_command_buffer_handle_t>(
      static_cast<size_t>(sizeof(ur_exp_command_buffer_desc_t)));
  if (descPtr) {
    reinterpret_cast<mock::dummy_handle_t>(*retCmdBuffer)
        ->setDataAs<ur_exp_command_buffer_desc_t>(*descPtr);
  }

  return UR_RESULT_SUCCESS;
}

inline ur_result_t mock_urCommandBufferGetInfoExp(void *pParams) {
  auto params =
      reinterpret_cast<ur_command_buffer_get_info_exp_params_t *>(pParams);

  auto cmdBufferDummyHandle =
      reinterpret_cast<mock::dummy_handle_t>(*params->phCommandBuffer);
  switch (*params->ppropName) {
  case UR_EXP_COMMAND_BUFFER_INFO_DESCRIPTOR: {
    if (*params->ppPropValue) {
      ur_exp_command_buffer_desc_t *propValue =
          reinterpret_cast<ur_exp_command_buffer_desc_t *>(
              *params->ppPropValue);
      *propValue =
          cmdBufferDummyHandle->getDataAs<ur_exp_command_buffer_desc_t>();
    }
    if (*params->ppPropSizeRet)
      **params->ppPropSizeRet = sizeof(ur_exp_command_buffer_desc_t);
  }
    return UR_RESULT_SUCCESS;
  default:
    return UR_RESULT_SUCCESS;
  }
  return UR_RESULT_SUCCESS;
}

// Checking command handle behaviour only
inline ur_result_t mock_urCommandBufferAppendKernelLaunchExp(void *pParams) {
  auto params =
      reinterpret_cast<ur_command_buffer_append_kernel_launch_exp_params_t *>(
          pParams);

  auto cmdBufferDummyHandle =
      reinterpret_cast<mock::dummy_handle_t>(*params->phCommandBuffer);
  // Requesting a command handle when the command buffer is not updatable is an
  // error
  if (*(params->pphCommand) &&
      cmdBufferDummyHandle->getDataAs<ur_exp_command_buffer_desc_t>()
              .isUpdatable == false) {
    return UR_RESULT_ERROR_INVALID_OPERATION;
  }

  return UR_RESULT_SUCCESS;
}

} // namespace MockAdapter

/// The UrMock<> class sets up UR for adapter mocking with the set of default
/// overrides above, and ensures the appropriate parts of the sycl runtime and
/// UR mocking code are reset/torn down in between tests.
///
/// The template parameter allows tests to select an arbitrary backend to have
/// the mock adapter report itself as.
template <sycl::backend Backend = backend::opencl> class UrMock {
public:
  /// Constructs UrMock<>
  ///
  /// This ensures UR is setup for adapter mocking and also injects our default
  /// entry-point overrides into the mock adapter.
  UrMock() {
#define ADD_DEFAULT_OVERRIDE(func_name, func_override)                         \
  mock::getCallbacks().set_replace_callback(#func_name,                        \
                                            &MockAdapter::func_override);
    ADD_DEFAULT_OVERRIDE(urAdapterGetInfo,
                         mock_urAdapterGetInfo<convertToUrBackend(Backend)>)
    ADD_DEFAULT_OVERRIDE(urPlatformGet, mock_urPlatformGet)
    ADD_DEFAULT_OVERRIDE(urDeviceGet, mock_urDeviceGet)
    ADD_DEFAULT_OVERRIDE(urDeviceRetain, mock_urDeviceRetain)
    ADD_DEFAULT_OVERRIDE(urDeviceRelease, mock_urDeviceRelease)
    ADD_DEFAULT_OVERRIDE(urPlatformGetInfo,
                         mock_urPlatformGetInfo<convertToUrBackend(Backend)>)
    ADD_DEFAULT_OVERRIDE(urDeviceGetInfo, mock_urDeviceGetInfo)
    ADD_DEFAULT_OVERRIDE(urProgramGetInfo, mock_urProgramGetInfo)
    ADD_DEFAULT_OVERRIDE(urContextGetInfo, mock_urContextGetInfo)
    ADD_DEFAULT_OVERRIDE(urQueueGetInfo, mock_urQueueGetInfo)
    ADD_DEFAULT_OVERRIDE(urProgramGetInfo, mock_urProgramGetInfo)
    ADD_DEFAULT_OVERRIDE(urKernelGetGroupInfo, mock_urKernelGetGroupInfo)
    ADD_DEFAULT_OVERRIDE(urEventGetInfo, mock_urEventGetInfo)
    ADD_DEFAULT_OVERRIDE(urKernelSuggestMaxCooperativeGroupCountExp,
                         mock_urKernelSuggestMaxCooperativeGroupCountExp)
    ADD_DEFAULT_OVERRIDE(urDeviceSelectBinary, mock_urDeviceSelectBinary)
    ADD_DEFAULT_OVERRIDE(urPlatformGetBackendOption,
                         mock_urPlatformGetBackendOption)
    ADD_DEFAULT_OVERRIDE(urDeviceGetGlobalTimestamps,
                         mock_urDeviceGetGlobalTimestamps)
    ADD_DEFAULT_OVERRIDE(urUsmP2PPeerAccessGetInfoExp,
                         mock_urUsmP2PPeerAccessGetInfoExp)
    ADD_DEFAULT_OVERRIDE(urVirtualMemReserve, mock_urVirtualMemReserve)
    ADD_DEFAULT_OVERRIDE(urCommandBufferCreateExp,
                         mock_urCommandBufferCreateExp);
    ADD_DEFAULT_OVERRIDE(urCommandBufferAppendKernelLaunchExp,
                         mock_urCommandBufferAppendKernelLaunchExp);
    ADD_DEFAULT_OVERRIDE(urCommandBufferGetInfoExp,
                         mock_urCommandBufferGetInfoExp);
#undef ADD_DEFAULT_OVERRIDE

    ur_loader_config_handle_t UrLoaderConfig = nullptr;

    urLoaderConfigCreate(&UrLoaderConfig);
    urLoaderConfigSetMockingEnabled(UrLoaderConfig, true);

    sycl::detail::ur::initializeUr(UrLoaderConfig);
    urLoaderConfigRelease(UrLoaderConfig);
  }

  UrMock(UrMock<Backend> &&Other) = delete;
  UrMock(const UrMock<Backend> &) = delete;
  UrMock<Backend> &operator=(const UrMock<Backend> &) = delete;
  ~UrMock() {
    // mock::getCallbacks() is an application lifetime object, we need to reset
    // these between tests
    detail::GlobalHandler::instance().prepareSchedulerToRelease(true);
    detail::GlobalHandler::instance().releaseDefaultContexts();
    // clear platform cache in case subsequent tests want a different backend,
    // this forces platforms to be reconstructed (and thus queries about UR
    // backend info to be called again)
    detail::GlobalHandler::instance().getPlatformCache().clear();
    mock::getCallbacks().resetCallbacks();
  }

private:
  // These two helpers are needed to enable arbitrary backend selection
  // at compile time.
  static constexpr ur_backend_t
  convertToUrBackend(const sycl::backend SyclBackend) {
    switch (SyclBackend) {
    case sycl::backend::opencl:
      return UR_BACKEND_OPENCL;
    case sycl::backend::ext_oneapi_level_zero:
      return UR_BACKEND_LEVEL_ZERO;
    case sycl::backend::ext_oneapi_cuda:
      return UR_BACKEND_CUDA;
    case sycl::backend::ext_oneapi_hip:
      return UR_BACKEND_HIP;
    case sycl::backend::ext_oneapi_native_cpu:
      return UR_BACKEND_NATIVE_CPU;
    default:
      return UR_BACKEND_UNKNOWN;
    }
  }
};

} // namespace unittest
} // namespace _V1
} // namespace sycl
