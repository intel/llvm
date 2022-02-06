//==------------- PiMock.cpp --- Mock unit testing library -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "helpers/PiMock.hpp"
#include "detail/global_handler.hpp"
#include "detail/plugin.hpp"

#include <CL/sycl/backend_types.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/pi.hpp>

#include <array>
#include <vector>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace unittest {

std::unordered_map<backend, PiDispatch> *GDispatchTables =
    new std::unordered_map<backend, PiDispatch>;

template <backend Backend> static detail::plugin getMockPlugin() {
  (*GDispatchTables)[Backend] = PiDispatch{};

  auto PIPlugin = std::make_shared<_pi_plugin>();

#define _PI_API(api)                                                           \
  PIPlugin->PiFunctionTable.api = [](auto... Args) {                           \
    return (*GDispatchTables)[Backend].mock_##api(Args...);                    \
  };
#include <CL/sycl/detail/pi.def>
#undef _PI_API

  detail::plugin Plugin{PIPlugin, Backend, nullptr};

  return Plugin;
}

void hijackPlugins() {
  // Force initialization with the only purposes to prevent runtime overriding
  // plugin replacements in future.
  RT::initialize();

  std::vector<detail::plugin> &Plugins =
      detail::GlobalHandler::instance().getPlugins();

  Plugins.clear();

  Plugins.push_back(getMockPlugin<backend::opencl>());
  Plugins.push_back(getMockPlugin<backend::ext_oneapi_level_zero>());
  Plugins.push_back(getMockPlugin<backend::ext_oneapi_cuda>());
  Plugins.push_back(getMockPlugin<backend::ext_oneapi_hip>());
  Plugins.push_back(getMockPlugin<backend::ext_intel_esimd_emulator>());
}

//----------------------------------------------------------------------------//
// Default API redefinitions
//----------------------------------------------------------------------------//

size_t *GPlatforms[5] = {new size_t{0}, new size_t{1}, new size_t{2},
                         new size_t{3}, new size_t{4}};

size_t *GDevices[5][3]{{new size_t{1}, new size_t{1}, new size_t{1}},
                       {new size_t{1}, new size_t{1}, new size_t{1}},
                       {new size_t{1}, new size_t{1}, new size_t{1}},
                       {new size_t{1}, new size_t{1}, new size_t{1}},
                       {new size_t{1}, new size_t{1}, new size_t{1}}};

static pi_result redefinedPlatformsGet(size_t idx, pi_uint32 num_entries,
                                       pi_platform *platforms,
                                       pi_uint32 *num_platforms) {
  if (num_platforms) {
    *num_platforms = 1;
  }
  if (platforms) {
    *platforms = reinterpret_cast<pi_platform>(GPlatforms[idx]);
  }

  return PI_SUCCESS;
}

static pi_result redefinedPlatformGetInfo(pi_platform platform,
                                          pi_platform_info param_name,
                                          size_t param_value_size,
                                          void *param_value,
                                          size_t *param_value_size_ret) {
  if (param_name == PI_PLATFORM_INFO_NAME) {
    std::string PN = "Mock Platform";
    if (param_value_size_ret) {
      *param_value_size_ret = PN.size() + 1;
    }
    if (param_value) {
      strncpy(static_cast<char *>(param_value), PN.data(), PN.size() + 1);
    }
  } else if (param_name == PI_PLATFORM_INFO_VERSION) {
    std::string PV = "OpenCL 2.1";
    if (param_value_size_ret) {
      *param_value_size_ret = PV.size() + 1;
    }
    if (param_value) {
      strncpy(static_cast<char *>(param_value), PV.data(), PV.size() + 1);
    }
  } else {
    std::cerr << "Unknown platform parameter " << std::hex << param_name
              << "\n";
    std::terminate();
  }

  return PI_SUCCESS;
}

static pi_result redefinedDevicesGet(pi_platform platform,
                                     pi_device_type device_type,
                                     pi_uint32 num_entries, pi_device *devices,
                                     pi_uint32 *num_devices) {
  if (num_devices) {
    *num_devices = 3;
  }

  if (devices) {
    size_t Idx = *reinterpret_cast<size_t *>(platform);
    for (size_t I = 0; I < 3; I++) {
      devices[I] = reinterpret_cast<pi_device>(GDevices[Idx][I]);
    }
  }

  return PI_SUCCESS;
}

static pi_result
redefinedDeviceGetNativeHandle(pi_device device,
                               pi_native_handle *nativeHandle) {
  assert(device);
  *nativeHandle = reinterpret_cast<pi_native_handle>(device);
  return PI_SUCCESS;
}

static pi_result
redefinedDeviceCreateWithNativeHandle(pi_native_handle nativeHandle,
                                      pi_platform platform, pi_device *device) {
  assert(nativeHandle);
  *device = reinterpret_cast<pi_device>(nativeHandle);
  return PI_SUCCESS;
}

static pi_result redefinedDeviceGetInfo(pi_device device,
                                        pi_device_info param_name,
                                        size_t param_value_size,
                                        void *param_value,
                                        size_t *param_value_size_ret) {
  if (param_name == PI_DEVICE_INFO_TYPE) {
    if (param_value_size_ret) {
      *param_value_size_ret = sizeof(pi_uint64);
    }
    if (param_value) {
      size_t Val = *reinterpret_cast<size_t *>(device);
      if (Val == 1) {
        *static_cast<pi_uint64 *>(param_value) = PI_DEVICE_TYPE_CPU;
      }
      if (Val == 2) {
        *static_cast<pi_uint64 *>(param_value) = PI_DEVICE_TYPE_GPU;
      }
      if (Val == 3) {
        *static_cast<pi_uint64 *>(param_value) = PI_DEVICE_TYPE_ACC;
      }
    }
  } else if (param_name == PI_DEVICE_INFO_NAME) {
    std::string DN = "Mock Device";
    if (param_value_size_ret) {
      *param_value_size_ret = DN.size() + 1;
    }
    if (param_value) {
      strncpy(static_cast<char *>(param_value), DN.data(), DN.size() + 1);
    }
  } else if (param_name == PI_DEVICE_INFO_VERSION) {
    std::string DV = "Mock Device 1.0";
    if (param_value_size_ret) {
      *param_value_size_ret = DV.size() + 1;
    }
    if (param_value) {
      strncpy(static_cast<char *>(param_value), DV.data(), DV.size() + 1);
    }
  } else if (param_name == PI_DEVICE_INFO_DRIVER_VERSION) {
    std::string DV = "0.0";
    if (param_value_size_ret) {
      *param_value_size_ret = DV.size() + 1;
    }
    if (param_value) {
      strncpy(static_cast<char *>(param_value), DV.data(), DV.size() + 1);
    }
  } else if (param_name == PI_DEVICE_INFO_PARENT_DEVICE) {
    if (param_value_size_ret) {
      *param_value_size_ret = sizeof(pi_device);
    }
    if (param_value) {
      *static_cast<void **>(param_value) = nullptr;
    }
  } else if (param_name == PI_DEVICE_INFO_EXTENSIONS) {
    std::string Extensions = "cl_khr_il_program";
    if (param_value_size_ret) {
      *param_value_size_ret = Extensions.size() + 1;
    }
    if (param_value) {
      strncpy(static_cast<char *>(param_value), Extensions.data(),
              Extensions.size() + 1);
    }
  } else if (param_name == PI_DEVICE_INFO_HOST_UNIFIED_MEMORY) {
    if (param_value_size_ret) {
      *param_value_size_ret = sizeof(pi_bool);
    }
    if (param_value) {
      *static_cast<pi_bool *>(param_value) = 1;
    }
  } else if (param_name == PI_DEVICE_INFO_COMPILER_AVAILABLE) {
    if (param_value_size_ret) {
      *param_value_size_ret = sizeof(pi_bool);
    }
    if (param_value) {
      *static_cast<pi_bool *>(param_value) = 1;
    }
  } else if (param_name == PI_DEVICE_INFO_LINKER_AVAILABLE) {
    if (param_value_size_ret) {
      *param_value_size_ret = sizeof(pi_bool);
    }
    if (param_value) {
      *static_cast<pi_bool *>(param_value) = 1;
    }
  } else {
    std::cerr << "Unsupported Device Info: " << std::hex << param_name << "\n";
    std::terminate();
  }
  return PI_SUCCESS;
}

static pi_result redefinedContextCreate(
    const pi_context_properties *properties, pi_uint32 num_devices,
    const pi_device *devices,
    void (*pfn_notify)(const char *errinfo, const void *private_info, size_t cb,
                       void *user_data),
    void *user_data, pi_context *ret_context) {
  *ret_context = reinterpret_cast<pi_context>(new size_t{0});

  return PI_SUCCESS;
}

static pi_result redefinedContextGetInfo(pi_context context,
                                         pi_context_info param_name,
                                         size_t param_value_size,
                                         void *param_value,
                                         size_t *param_value_size_ret) {
  std::cerr << "Unsupported context info " << std::hex << param_name << "\n";
  return PI_SUCCESS;
}

static pi_result redefinedContextRetain(pi_context) { return PI_SUCCESS; }

static pi_result redefinedContextRelease(pi_context) { return PI_SUCCESS; }

static pi_result redefinedContextGetNativeHandle(pi_context Ctx,
                                                 pi_native_handle *Handle) {
  *Handle = reinterpret_cast<pi_native_handle>(Ctx);
  return PI_SUCCESS;
}

static pi_result redefinedContextCreateWithNativeHandle(
    pi_native_handle nativeHandle, pi_uint32 numDevices,
    const pi_device *devices, bool pluginOwnsNativeHandle,
    pi_context *context) {
  *context = reinterpret_cast<pi_context>(nativeHandle);
  return PI_SUCCESS;
}

static pi_result redefinedQueueCreate(pi_context context, pi_device device,
                                      pi_queue_properties properties,
                                      pi_queue *queue) {
  *queue = reinterpret_cast<pi_queue>(new size_t{1});
  return PI_SUCCESS;
}

static pi_result redefinedQueueGetNativeHandle(pi_queue queue,
                                               pi_native_handle *nativeHandle) {
  *nativeHandle = reinterpret_cast<pi_native_handle>(queue);
  return PI_SUCCESS;
}

static pi_result
redefinedQueueCreateWithNativeHandle(pi_native_handle nativeHandle,
                                     pi_context context, pi_queue *queue,
                                     bool pluginOwnsNativeHandle) {
  *queue = reinterpret_cast<pi_queue>(nativeHandle);
  return PI_SUCCESS;
}

static pi_result redefinedQueueGetInfo(pi_queue command_queue,
                                       pi_queue_info param_name,
                                       size_t param_value_size,
                                       void *param_value,
                                       size_t *param_value_size_ret) {
  std::cerr << "Unknown queue info " << std::hex << param_name << "\n";
  return PI_SUCCESS;
}

static pi_result redefinedQueueRetain(pi_queue command_queue) {
  return PI_SUCCESS;
}

static pi_result redefinedQueueRelease(pi_queue command_queue) {
  return PI_SUCCESS;
}

static pi_result redefinedQueueFinish(pi_queue command_queue) {
  return PI_SUCCESS;
}

static pi_result redefinedQueueFlush(pi_queue command_queue) {
  return PI_SUCCESS;
}

static pi_result redefinedUSMHostAlloc(void **result_ptr, pi_context context,
                                       pi_usm_mem_properties *properties,
                                       size_t size, pi_uint32 alignment) {
  *result_ptr = new char[size];
  return PI_SUCCESS;
}

static pi_result redefinedUSMDeviceSharedAlloc(
    void **result_ptr, pi_context context, pi_device device,
    pi_usm_mem_properties *properties, size_t size, pi_uint32 alignment) {
  *result_ptr = new char[size];
  return PI_SUCCESS;
}

static pi_result redefinedUSMFree(pi_context context, void *ptr) {
  delete[] static_cast<char *>(ptr);
  return PI_SUCCESS;
}

static pi_result
redefinedMemBufferCreate(pi_context context, pi_mem_flags flags, size_t size,
                         void *host_ptr, pi_mem *ret_mem,
                         const pi_mem_properties *properties = nullptr) {
  *ret_mem = reinterpret_cast<pi_mem>(new size_t{size});

  return PI_SUCCESS;
}

static pi_result redefinedMemRetain(pi_mem mem) { return PI_SUCCESS; }

static pi_result redefinedMemRelease(pi_mem mem) { return PI_SUCCESS; }

inline pi_result redefinedProgramCreateCommon(pi_context, const void *, size_t,
                                              pi_program *ret_program) {
  *ret_program = reinterpret_cast<pi_program>(1);
  return PI_SUCCESS;
}

static pi_result
redefinedProgramGetNativeHandle(pi_program program,
                                pi_native_handle *nativeHandle) {
  *nativeHandle = reinterpret_cast<pi_native_handle>(program);
  return PI_SUCCESS;
}

static pi_result redefinedProgramCreateWithNativeHandle(
    pi_native_handle nativeHandle, pi_context context,
    bool pluginOwnsNativeHandle, pi_program *program) {
  *program = reinterpret_cast<pi_program>(nativeHandle);
  return PI_SUCCESS;
}

inline pi_result
redefinedProgramCreateWithBinary(pi_context, pi_uint32, const pi_device *,
                                 const size_t *, const unsigned char **, size_t,
                                 const pi_device_binary_property *, pi_int32 *,
                                 pi_program *ret_program) {
  *ret_program = reinterpret_cast<pi_program>(1);
  return PI_SUCCESS;
}

static pi_result redefinedProgramBuildCommon(
    pi_program prog, pi_uint32, const pi_device *, const char *,
    void (*pfn_notify)(pi_program program, void *user_data), void *user_data) {
  using namespace std::literals::chrono_literals;
  std::this_thread::sleep_for(100ns);
  if (pfn_notify) {
    pfn_notify(prog, user_data);
  }
  return PI_SUCCESS;
}

static pi_result redefinedProgramCompileCommon(
    pi_program, pi_uint32, const pi_device *, const char *, pi_uint32,
    const pi_program *, const char **, void (*)(pi_program, void *), void *) {
  return PI_SUCCESS;
}

inline pi_result redefinedProgramLinkCommon(pi_context, pi_uint32,
                                            const pi_device *, const char *,
                                            pi_uint32, const pi_program *,
                                            void (*)(pi_program, void *),
                                            void *, pi_program *ret_program) {
  *ret_program = reinterpret_cast<pi_program>(1);
  return PI_SUCCESS;
}

inline pi_result redefinedProgramGetInfoCommon(pi_program program,
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

static pi_result redefinedKernelCreateCommon(pi_program program,
                                             const char *kernel_name,
                                             pi_kernel *ret_kernel) {
  *ret_kernel = reinterpret_cast<pi_kernel>(1);
  return PI_SUCCESS;
}

static pi_result redefinedKernelCreateWithNativeHandle(
    pi_native_handle nativeHandle, pi_context context, pi_program program,
    bool pluginOwnsNativeHandle, pi_kernel *kernel) {
  *kernel = reinterpret_cast<pi_kernel>(nativeHandle);
  return PI_SUCCESS;
}

static pi_result
redefinedKernelGetNativeHandle(pi_kernel kernel,
                               pi_native_handle *nativeHandle) {
  *nativeHandle = reinterpret_cast<pi_native_handle>(kernel);
  return PI_SUCCESS;
}

inline pi_result redefinedKernelGetInfoCommon(pi_kernel kernel,
                                              pi_kernel_info param_name,
                                              size_t param_value_size,
                                              void *param_value,
                                              size_t *param_value_size_ret) {
  return PI_SUCCESS;
}

inline pi_result redefinedKernelSetExecInfoCommon(
    pi_kernel kernel, pi_kernel_exec_info value_name, size_t param_value_size,
    const void *param_value) {
  return PI_SUCCESS;
}

static pi_result redefinedEventGetInfoCommon(pi_event event,
                                             pi_event_info param_name,
                                             size_t param_value_size,
                                             void *param_value,
                                             size_t *param_value_size_ret) {
  if (param_name == PI_EVENT_INFO_COMMAND_EXECUTION_STATUS) {
    auto *status = reinterpret_cast<pi_event_status *>(param_value);
    *status = PI_EVENT_SUBMITTED;
  }
  return PI_SUCCESS;
}

static pi_result redefinedEventGetNativeHandle(pi_event event,
                                               pi_native_handle *nativeHandle) {
  *nativeHandle = reinterpret_cast<pi_native_handle>(event);
  return PI_SUCCESS;
}

static pi_result
redefinedEventCreateWithNativeHandle(pi_native_handle nativeHandle,
                                     pi_context context, bool ownNativeHandle,
                                     pi_event *event) {
  *event = reinterpret_cast<pi_event>(nativeHandle);
  return PI_SUCCESS;
}

inline pi_result redefinedEnqueueKernelLaunchCommon(
    pi_queue, pi_kernel, pi_uint32, const size_t *, const size_t *,
    const size_t *, pi_uint32, const pi_event *, pi_event *event) {
  *event = reinterpret_cast<pi_event>(new int{});
  return PI_SUCCESS;
}

inline pi_result redefinedKernelGetGroupInfoCommon(
    pi_kernel kernel, pi_device device, pi_kernel_group_info param_name,
    size_t param_value_size, void *param_value, size_t *param_value_size_ret) {
  if (param_name == PI_KERNEL_GROUP_INFO_WORK_GROUP_SIZE && param_value) {
    auto RealVal = reinterpret_cast<size_t *>(param_value);
    RealVal[0] = 0;
    RealVal[1] = 0;
    RealVal[2] = 0;
  }
  return PI_SUCCESS;
}
inline pi_result redefinedDeviceSelectBinary(pi_device device,
                                             pi_device_binary *binaries,
                                             pi_uint32 num_binaries,
                                             pi_uint32 *selected_binary_ind) {
  *selected_binary_ind = 0;
  return PI_SUCCESS;
}

constexpr auto NOP = [](auto...) { return PI_SUCCESS; };

void setupDefaultMockAPIs() {
  using namespace sycl::detail;
  using namespace sycl::unittest;

  redefineOne<PiApiKind::piPlatformsGet, backend::opencl>(
      [](pi_uint32 num_entries, pi_platform *platforms,
         pi_uint32 *num_platforms) {
        return redefinedPlatformsGet(0, num_entries, platforms, num_platforms);
      });
  redefineOne<PiApiKind::piPlatformsGet, backend::ext_oneapi_level_zero>(
      [](pi_uint32 num_entries, pi_platform *platforms,
         pi_uint32 *num_platforms) {
        return redefinedPlatformsGet(1, num_entries, platforms, num_platforms);
      });
  redefineOne<PiApiKind::piPlatformsGet, backend::ext_oneapi_cuda>(
      [](pi_uint32 num_entries, pi_platform *platforms,
         pi_uint32 *num_platforms) {
        return redefinedPlatformsGet(2, num_entries, platforms, num_platforms);
      });
  redefineOne<PiApiKind::piPlatformsGet, backend::ext_oneapi_hip>(
      [](pi_uint32 num_entries, pi_platform *platforms,
         pi_uint32 *num_platforms) {
        return redefinedPlatformsGet(3, num_entries, platforms, num_platforms);
      });
  redefineOne<PiApiKind::piPlatformsGet, backend::ext_intel_esimd_emulator>(
      [](pi_uint32 num_entries, pi_platform *platforms,
         pi_uint32 *num_platforms) {
        return redefinedPlatformsGet(4, num_entries, platforms, num_platforms);
      });
  redefine<PiApiKind::piPlatformGetInfo>(redefinedPlatformGetInfo);
  redefine<PiApiKind::piDevicesGet>(redefinedDevicesGet);
  redefine<PiApiKind::piextDeviceGetNativeHandle>(
      redefinedDeviceGetNativeHandle);
  redefine<PiApiKind::piextDeviceCreateWithNativeHandle>(
      redefinedDeviceCreateWithNativeHandle);
  redefine<PiApiKind::piDeviceGetInfo>(redefinedDeviceGetInfo);
  redefine<PiApiKind::piDeviceRetain>(NOP);
  redefine<PiApiKind::piDeviceRelease>(NOP);
  redefine<PiApiKind::piContextCreate>(redefinedContextCreate);
  redefine<PiApiKind::piContextGetInfo>(redefinedContextGetInfo);
  redefine<PiApiKind::piContextRelease>(redefinedContextRelease);
  redefine<PiApiKind::piContextRetain>(redefinedContextRetain);
  redefine<PiApiKind::piextContextGetNativeHandle>(
      redefinedContextGetNativeHandle);
  redefine<PiApiKind::piextContextCreateWithNativeHandle>(
      redefinedContextCreateWithNativeHandle);
  redefine<PiApiKind::piQueueCreate>(redefinedQueueCreate);
  redefine<PiApiKind::piQueueGetInfo>(redefinedQueueGetInfo);
  redefine<PiApiKind::piQueueRelease>(redefinedQueueRelease);
  redefine<PiApiKind::piQueueRetain>(redefinedQueueRetain);
  redefine<PiApiKind::piQueueFinish>(redefinedQueueFinish);
  redefine<PiApiKind::piQueueFlush>(redefinedQueueFlush);
  redefine<PiApiKind::piextQueueGetNativeHandle>(redefinedQueueGetNativeHandle);
  redefine<PiApiKind::piextQueueCreateWithNativeHandle>(
      redefinedQueueCreateWithNativeHandle);
  redefine<PiApiKind::piMemBufferCreate>(redefinedMemBufferCreate);
  redefine<PiApiKind::piMemRelease>(redefinedMemRelease);
  redefine<PiApiKind::piMemRetain>(redefinedMemRetain);
  redefine<PiApiKind::piextUSMHostAlloc>(redefinedUSMHostAlloc);
  redefine<PiApiKind::piextUSMDeviceAlloc>(redefinedUSMDeviceSharedAlloc);
  redefine<PiApiKind::piextUSMDeviceAlloc>(redefinedUSMDeviceSharedAlloc);
  redefine<PiApiKind::piextUSMFree>(redefinedUSMFree);
  redefine<PiApiKind::piEventRelease>(NOP);
  redefine<PiApiKind::piEventsWait>(NOP);
  redefine<PiApiKind::piEventGetInfo>(redefinedEventGetInfoCommon);
  redefine<PiApiKind::piextEventGetNativeHandle>(redefinedEventGetNativeHandle);
  redefine<PiApiKind::piextEventCreateWithNativeHandle>(
      redefinedEventCreateWithNativeHandle);
  redefine<PiApiKind::piProgramCreate>(redefinedProgramCreateCommon);
  redefine<PiApiKind::piProgramCreateWithBinary>(
      redefinedProgramCreateWithBinary);
  redefine<PiApiKind::piProgramCompile>(redefinedProgramCompileCommon);
  redefine<PiApiKind::piProgramLink>(redefinedProgramLinkCommon);
  redefine<PiApiKind::piProgramBuild>(redefinedProgramBuildCommon);
  redefine<PiApiKind::piProgramGetInfo>(redefinedProgramGetInfoCommon);
  redefine<PiApiKind::piProgramRetain>(NOP);
  redefine<PiApiKind::piProgramRelease>(NOP);
  redefine<PiApiKind::piextProgramGetNativeHandle>(
      redefinedProgramGetNativeHandle);
  redefine<PiApiKind::piextProgramCreateWithNativeHandle>(
      redefinedProgramCreateWithNativeHandle);
  redefine<PiApiKind::piKernelCreate>(redefinedKernelCreateCommon);
  redefine<PiApiKind::piKernelRetain>(NOP);
  redefine<PiApiKind::piKernelRelease>(NOP);
  redefine<PiApiKind::piKernelGetInfo>(redefinedKernelGetInfoCommon);
  redefine<PiApiKind::piextKernelCreateWithNativeHandle>(
      redefinedKernelCreateWithNativeHandle);
  redefine<PiApiKind::piextKernelGetNativeHandle>(
      redefinedKernelGetNativeHandle);
  redefine<PiApiKind::piKernelGetGroupInfo>(redefinedKernelGetGroupInfoCommon);
  redefine<PiApiKind::piKernelSetExecInfo>(redefinedKernelSetExecInfoCommon);
  redefine<PiApiKind::piEnqueueKernelLaunch>(
      redefinedEnqueueKernelLaunchCommon);
  redefine<PiApiKind::piextDeviceSelectBinary>(redefinedDeviceSelectBinary);
}

void resetMockAPIs() {
  (*GDispatchTables)[backend::opencl] = PiDispatch{};
  (*GDispatchTables)[backend::ext_oneapi_level_zero] = PiDispatch{};
  (*GDispatchTables)[backend::ext_oneapi_cuda] = PiDispatch{};
  (*GDispatchTables)[backend::ext_oneapi_hip] = PiDispatch{};
  (*GDispatchTables)[backend::ext_intel_esimd_emulator] = PiDispatch{};

  redefine<detail::PiApiKind::piEventsWait>(NOP);
  redefine<detail::PiApiKind::piEventRelease>(NOP);
  redefine<detail::PiApiKind::piDeviceRelease>(NOP);
  redefine<detail::PiApiKind::piContextRelease>(NOP);
  redefine<detail::PiApiKind::piKernelRetain>(NOP);
  redefine<detail::PiApiKind::piKernelRelease>(NOP);
  redefine<detail::PiApiKind::piProgramRetain>(NOP);
  redefine<detail::PiApiKind::piProgramRelease>(NOP);
  redefine<detail::PiApiKind::piTearDown>(NOP);
}
} // namespace unittest
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
