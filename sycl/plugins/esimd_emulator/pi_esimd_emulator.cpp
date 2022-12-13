//===---------- pi_esimd_emulator.cpp - CM Emulation Plugin ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file pi_esimd_emulator.cpp
/// Declarations for CM Emulation Plugin. It is the interface between the
/// device-agnostic SYCL runtime layer and underlying CM Emulation
///
/// \ingroup sycl_pi_esimd_emulator

#include <stdint.h>

#include <detail/accessor_impl.hpp>
#include <sycl/backend_types.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/export.hpp>
#include <sycl/detail/helpers.hpp>
#include <sycl/detail/host_profiling_info.hpp>
#include <sycl/detail/kernel_desc.hpp>
#include <sycl/detail/type_traits.hpp>
#include <sycl/ext/intel/esimd/common.hpp> // SLM_BTI
#include <sycl/group.hpp>
#include <sycl/id.hpp>
#include <sycl/kernel.hpp>
#include <sycl/nd_item.hpp>
#include <sycl/range.hpp>

#include <esimdemu_support.h>

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include "pi_esimd_emulator.hpp"

#define ARG_UNUSED(x) (void)x

namespace {

// Helper functions for unified 'Return' type declaration - imported
// from pi_level_zero.cpp
template <typename T, typename Assign>
pi_result getInfoImpl(size_t ParamValueSize, void *ParamValue,
                      size_t *ParamValueSizeRet, T Value, size_t ValueSize,
                      Assign &&AssignFunc) {
  if (ParamValue != nullptr) {
    if (ParamValueSize < ValueSize) {
      return PI_ERROR_INVALID_VALUE;
    }
    AssignFunc(ParamValue, Value, ValueSize);
  }
  if (ParamValueSizeRet != nullptr) {
    *ParamValueSizeRet = ValueSize;
  }
  return PI_SUCCESS;
}

template <typename T>
pi_result getInfo(size_t ParamValueSize, void *ParamValue,
                  size_t *ParamValueSizeRet, T Value) {
  auto assignment = [](void *ParamValue, T Value, size_t ValueSize) {
    ARG_UNUSED(ValueSize);
    *static_cast<T *>(ParamValue) = Value;
  };
  return getInfoImpl(ParamValueSize, ParamValue, ParamValueSizeRet, Value,
                     sizeof(T), assignment);
}

template <typename T>
pi_result getInfoArray(size_t ArrayLength, size_t ParamValueSize,
                       void *ParamValue, size_t *ParamValueSizeRet, T *Value) {
  return getInfoImpl(ParamValueSize, ParamValue, ParamValueSizeRet, Value,
                     ArrayLength * sizeof(T), memcpy);
}

template <>
pi_result getInfo<const char *>(size_t ParamValueSize, void *ParamValue,
                                size_t *ParamValueSizeRet, const char *Value) {
  return getInfoArray(strlen(Value) + 1, ParamValueSize, ParamValue,
                      ParamValueSizeRet, Value);
}

class ReturnHelper {
public:
  ReturnHelper(size_t ArgParamValueSize, void *ArgParamValue,
               size_t *ArgParamValueSizeRet)
      : ParamValueSize(ArgParamValueSize), ParamValue(ArgParamValue),
        ParamValueSizeRet(ArgParamValueSizeRet) {}

  template <class T> pi_result operator()(const T &t) {
    return getInfo(ParamValueSize, ParamValue, ParamValueSizeRet, t);
  }

private:
  size_t ParamValueSize;
  void *ParamValue;
  size_t *ParamValueSizeRet;
};

} // anonymous namespace

// Controls PI level tracing prints.
static bool PrintPiTrace = false;

static void PiTrace(std::string TraceString) {
  if (PrintPiTrace) {
    std::cout << TraceString << std::endl;
  }
}

// Global variables used in PI_esimd_emulator
// Note we only create a simple pointer variables such that C++ RT won't
// deallocate them automatically at the end of the main program.
// The heap memory allocated for this global variable reclaimed only when
// Sycl RT calls piTearDown().
static sycl::detail::ESIMDEmuPluginOpaqueData *PiESimdDeviceAccess;

// Single-entry cache for piPlatformsGet call.
static pi_platform PiPlatformCache;
// TODO/FIXME : Memory leak. Handle with 'piTearDown'.
static std::mutex *PiPlatformCacheLock = new std::mutex;

// Mapping between surface index and CM-managed surface
static std::unordered_map<unsigned int, _pi_mem *> *PiESimdSurfaceMap =
    new std::unordered_map<unsigned int, _pi_mem *>;
// TODO/FIXME : Memory leak. Handle with 'piTearDown'.
static std::mutex *PiESimdSurfaceMapLock = new std::mutex;

// To be compared with ESIMD_EMULATOR_PLUGIN_OPAQUE_DATA_VERSION in device
// interface header file
#define ESIMDEmuPluginDataVersion 0

// To be compared with ESIMD_DEVICE_INTERFACE_VERSION in device
// interface header file
#define ESIMDEmuPluginInterfaceVersion 1

// For PI_DEVICE_INFO_DRIVER_VERSION info
static char ESimdEmuVersionString[32];

// Global variables for PI_ERROR_PLUGIN_SPECIFIC_ERROR
constexpr size_t MaxMessageSize = 256;
thread_local pi_result ErrorMessageCode = PI_SUCCESS;
thread_local char ErrorMessage[MaxMessageSize];

// Utility function for setting a message and warning
[[maybe_unused]] static void setErrorMessage(const char *message,
                                             pi_result error_code) {
  assert(strlen(message) <= MaxMessageSize);
  strcpy(ErrorMessage, message);
  ErrorMessageCode = error_code;
}

// Returns plugin specific error and warning messages
pi_result piPluginGetLastError(char **message) {
  *message = &ErrorMessage[0];
  return ErrorMessageCode;
}

using IDBuilder = sycl::detail::Builder;

template <int NDims>
using KernelFunc = std::function<void(const sycl::nd_item<NDims> &)>;

// Struct to wrap dimension info and lambda function to be invoked by
// CM Kernel launcher that only accepts raw function pointer for
// kernel execution. Function instances of 'InvokeKernel' un-wrap
// this struct instance and invoke lambda function ('Func')
template <int NDims> struct KernelInvocationContext {
  KernelFunc<NDims> Func;
  const sycl::range<NDims> &LocalSize;
  const sycl::range<NDims> &GlobalSize;
  const sycl::id<NDims> &GlobalOffset;
};

// A helper structure to create multi-dimensional range when
// dimensionality is given as a template parameter. `create` function
// in specializations accepts a template `Gen` function which
// generates range extent for a dimension given as an argument.
template <int NDims> struct RangeBuilder;

template <> struct RangeBuilder<1> {
  template <typename Gen> static sycl::range<1> create(Gen G) {
    return sycl::range<1>{G(0)};
  }
};
template <> struct RangeBuilder<2> {
  template <typename Gen> static sycl::range<2> create(Gen G) {
    return sycl::range<2>{G(0), G(1)};
  }
};
template <> struct RangeBuilder<3> {
  template <typename Gen> static sycl::range<3> create(Gen G) {
    return sycl::range<3>{G(0), G(1), G(2)};
  }
};

// Function template to generate entry point of kernel execution as
// raw function pointer. CM kernel launcher executes one instance of
// this function per 'NDims'
template <int NDims> void InvokeKernel(KernelInvocationContext<NDims> *ctx) {

  sycl::range<NDims> GroupSize{
      sycl::detail::InitializedVal<NDims, sycl::range>::template get<0>()};

  for (int i = 0; i < NDims; ++i) {
    GroupSize[i] = ctx->GlobalSize[i] / ctx->LocalSize[i];
  }

  const sycl::id<NDims> LocalID = RangeBuilder<NDims>::create(
      [](int i) { return cm_support::get_thread_idx(i); });

  const sycl::id<NDims> GroupID = RangeBuilder<NDims>::create(
      [](int i) { return cm_support::get_group_idx(i); });

  const sycl::group<NDims> Group = IDBuilder::createGroup<NDims>(
      ctx->GlobalSize, ctx->LocalSize, GroupSize, GroupID);

  const sycl::id<NDims> GlobalID =
      GroupID * ctx->LocalSize + LocalID + ctx->GlobalOffset;

  const sycl::item<NDims, /*Offset=*/true> GlobalItem =
      IDBuilder::createItem<NDims, true>(ctx->GlobalSize, GlobalID,
                                         ctx->GlobalOffset);

  const sycl::item<NDims, /*Offset=*/false> LocalItem =
      IDBuilder::createItem<NDims, false>(ctx->LocalSize, LocalID);

  const sycl::nd_item<NDims> NDItem =
      IDBuilder::createNDItem<NDims>(GlobalItem, LocalItem, Group);

  ctx->Func(NDItem);
}

// Interface for lauching kernels using libcm from CM EMU project.
template <int DIMS> class libCMBatch {
private:
  const KernelFunc<DIMS> &MKernel;
  std::vector<uint32_t> GroupDim, SpaceDim;

public:
  libCMBatch(const KernelFunc<DIMS> &Kernel)
      : MKernel(Kernel), GroupDim{1, 1, 1}, SpaceDim{1, 1, 1} {}

  void runIterationSpace(const sycl::range<DIMS> &LocalSize,
                         const sycl::range<DIMS> &GlobalSize,
                         const sycl::id<DIMS> &GlobalOffset) {

    for (int I = 0; I < DIMS; I++) {
      SpaceDim[I] = (uint32_t)LocalSize[I];
      GroupDim[I] = (uint32_t)(GlobalSize[I] / LocalSize[I]);
    }

    const auto InvokeKernelArg = KernelInvocationContext<DIMS>{
        MKernel, LocalSize, GlobalSize, GlobalOffset};

    EsimdemuKernel{reinterpret_cast<fptrVoid>(InvokeKernel<DIMS>),
                   GroupDim.data(), SpaceDim.data()}
        .launchMT(sizeof(InvokeKernelArg), &InvokeKernelArg);
  }
};

unsigned int sycl_get_cm_surface_index(void *PtrInput) {
  _pi_mem *Surface = static_cast<_pi_mem *>(PtrInput);

  return Surface->SurfaceIndex;
}

// Function to provide image info for kernel compilation using surface
// index without dependency on '_pi_image' definition
void sycl_get_cm_buffer_params(unsigned int IndexInput, char **BaseAddr,
                               uint32_t *Width, std::mutex **BufMtxLock) {
  std::lock_guard<std::mutex> Lock{*PiESimdSurfaceMapLock};
  auto MemIter = PiESimdSurfaceMap->find(IndexInput);

  assert(MemIter != PiESimdSurfaceMap->end() && "Invalid Surface Index");

  _pi_buffer *Buf = static_cast<_pi_buffer *>(MemIter->second);

  *BaseAddr = Buf->MapHostPtr;
  *Width = static_cast<uint32_t>(Buf->Size);

  *BufMtxLock = &(Buf->SurfaceLock);
}

// Function to provide image info for kernel compilation using surface
// index without dependency on '_pi_image' definition
void sycl_get_cm_image_params(unsigned int IndexInput, char **BaseAddr,
                              uint32_t *Width, uint32_t *Height, uint32_t *Bpp,
                              std::mutex **ImgMtxLock) {
  std::lock_guard<std::mutex> Lock{*PiESimdSurfaceMapLock};
  auto MemIter = PiESimdSurfaceMap->find(IndexInput);
  assert(MemIter != PiESimdSurfaceMap->end() && "Invalid Surface Index");

  _pi_image *Img = static_cast<_pi_image *>(MemIter->second);

  *BaseAddr = Img->MapHostPtr;

  *Bpp = static_cast<uint32_t>(Img->BytesPerPixel);
  *Width = static_cast<uint32_t>(Img->Width) * (*Bpp);
  *Height = static_cast<uint32_t>(Img->Height);

  *ImgMtxLock = &(Img->SurfaceLock);
}

/// Implementation for ESIMD_EMULATOR device interface accessing ESIMD
/// intrinsics and LibCM functionalties requred by intrinsics
sycl::detail::ESIMDDeviceInterface::ESIMDDeviceInterface() {
  version = ESIMDEmuPluginInterfaceVersion;
  reserved = nullptr;

  /* From 'esimd_emulator_functions_v1.h' : Start */
  cm_barrier_ptr = cm_support::barrier;
  cm_sbarrier_ptr = cm_support::split_barrier;
  cm_fence_ptr = cm_support::fence;

  sycl_get_surface_base_addr_ptr = cm_support::get_surface_base_addr;
  __cm_emu_get_slm_ptr = cm_support::get_slm_base;
  cm_slm_init_ptr = cm_support::init_slm;

  sycl_get_cm_surface_index_ptr = sycl_get_cm_surface_index;
  sycl_get_cm_buffer_params_ptr = sycl_get_cm_buffer_params;
  sycl_get_cm_image_params_ptr = sycl_get_cm_image_params;

  /* From 'esimd_emulator_functions_v1.h' : End */
}

/// Implementation for Host Kernel Launch used by
/// piEnqueueKernelLaunch

static bool isNull(int NDims, const size_t *R) {
  return ((0 == R[0]) && (NDims < 2 || 0 == R[1]) && (NDims < 3 || 0 == R[2]));
}

// NDims is the number of dimensions in the ND-range. Kernels are
// normalized in the handler so that all kernels take an sycl::nd_item
// as argument (see StoreLambda in sycl/handler.hpp). For kernels
// whose workgroup size (LocalWorkSize) is unspecified, InvokeImpl
// sets LocalWorkSize to {1, 1, 1}, i.e. each workgroup contains just
// one work item. CM emulator will run several workgroups in parallel
// depending on environment settings.

template <int NDims> struct InvokeImpl {

  static sycl::range<NDims> get_range(const size_t *Array) {
    if constexpr (NDims == 1)
      return sycl::range<NDims>{Array[0]};
    else if constexpr (NDims == 2)
      return sycl::range<NDims>{Array[0], Array[1]};
    else if constexpr (NDims == 3)
      return sycl::range<NDims>{Array[0], Array[1], Array[2]};
  }

  static void invoke(pi_kernel Kernel, const size_t *GlobalWorkOffset,
                     const size_t *GlobalWorkSize,
                     const size_t *LocalWorkSize) {
    libCMBatch<NDims>{*reinterpret_cast<KernelFunc<NDims> *>(Kernel)}
        .runIterationSpace(get_range(LocalWorkSize), get_range(GlobalWorkSize),
                           sycl::id<NDims>{get_range(GlobalWorkOffset)});
  }
};

extern "C" {

#define DIE_NO_IMPLEMENTATION                                                  \
  if (PrintPiTrace) {                                                          \
    std::cerr << "Not Implemented : " << __FUNCTION__                          \
              << " - File : " << __FILE__;                                     \
    std::cerr << " / Line : " << __LINE__ << std::endl;                        \
  }                                                                            \
  return PI_ERROR_INVALID_OPERATION;

#define CONTINUE_NO_IMPLEMENTATION                                             \
  if (PrintPiTrace) {                                                          \
    std::cerr << "Warning : Not Implemented : " << __FUNCTION__                \
              << " - File : " << __FILE__;                                     \
    std::cerr << " / Line : " << __LINE__ << std::endl;                        \
  }                                                                            \
  return PI_SUCCESS;

#define CASE_PI_UNSUPPORTED(not_supported)                                     \
  case not_supported:                                                          \
    if (PrintPiTrace) {                                                        \
      std::cerr << std::endl                                                   \
                << "Unsupported PI case : " << #not_supported << " in "        \
                << __FUNCTION__ << ":" << __LINE__ << "(" << __FILE__ << ")"   \
                << std::endl;                                                  \
    }                                                                          \
    return PI_ERROR_INVALID_OPERATION;

pi_result piPlatformsGet(pi_uint32 NumEntries, pi_platform *Platforms,
                         pi_uint32 *NumPlatforms) {
  static bool PiPlatformCachePopulated = false;
  static const char *PiTraceEnv = std::getenv("SYCL_PI_TRACE");
  static const int PiTraceValue = PiTraceEnv ? std::stoi(PiTraceEnv) : 0;

  if (PiTraceValue == -1) { // Means print all PI traces
    PrintPiTrace = true;
  }

  if (NumPlatforms) {
    *NumPlatforms = 1;
  }

  if (NumEntries == 0) {
    /// Runtime queries number of Platforms
    if (Platforms != nullptr) {
      PiTrace("Invalid Arguments for piPlatformsGet of "
              "esimd_emulator (Platforms!=nullptr) "
              "while querying number of platforms");
      return PI_ERROR_INVALID_VALUE;
    }
    return PI_SUCCESS;
  }

  if (Platforms == nullptr && NumPlatforms == nullptr) {
    return PI_ERROR_INVALID_VALUE;
  }

  std::lock_guard<std::mutex> Lock{*PiPlatformCacheLock};
  if (!PiPlatformCachePopulated) {
    PiPlatformCache = new _pi_platform();
    PiPlatformCache->CmEmuVersion = std::string("0.0.1");
    PiPlatformCachePopulated = true;
  }

  if (Platforms && NumEntries > 0) {
    *Platforms = PiPlatformCache;
  }

  return PI_SUCCESS;
}

pi_result piPlatformGetInfo(pi_platform Platform, pi_platform_info ParamName,
                            size_t ParamValueSize, void *ParamValue,
                            size_t *ParamValueSizeRet) {
  if (Platform == nullptr) {
    return PI_ERROR_INVALID_PLATFORM;
  }
  ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);

  switch (ParamName) {
  case PI_PLATFORM_INFO_NAME:
    return ReturnValue("Intel(R) ESIMD_EMULATOR/GPU");

  case PI_PLATFORM_INFO_VENDOR:
    return ReturnValue("Intel(R) Corporation");

  case PI_PLATFORM_INFO_VERSION:
    return ReturnValue(Platform->CmEmuVersion.c_str());

  case PI_PLATFORM_INFO_PROFILE:
    return ReturnValue("FULL_PROFILE");

  case PI_PLATFORM_INFO_EXTENSIONS:
    return ReturnValue("");

  default:
    // TODO: implement other parameters
    die("Unsupported ParamName in piPlatformGetInfo");
  }

  return PI_SUCCESS;
}

pi_result piextPlatformGetNativeHandle(pi_platform, pi_native_handle *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextPlatformCreateWithNativeHandle(pi_native_handle, pi_platform *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piDevicesGet(pi_platform Platform, pi_device_type DeviceType,
                       pi_uint32 NumEntries, pi_device *Devices,
                       pi_uint32 *NumDevices) {
  if (Platform == nullptr) {
    return PI_ERROR_INVALID_PLATFORM;
  }

  pi_result Res = Platform->populateDeviceCacheIfNeeded();
  if (Res != PI_SUCCESS) {
    return Res;
  }

  // CM has single-root-GPU-device without sub-device support.
  pi_uint32 DeviceCount = (DeviceType & PI_DEVICE_TYPE_GPU) ? 1 : 0;

  if (NumDevices) {
    *NumDevices = DeviceCount;
  }

  if (NumEntries == 0) {
    /// Runtime queries number of devices
    if (Devices != nullptr) {
      PiTrace("Invalid Arguments for piDevicesGet of esimd_emultor "
              "(Devices!=nullptr) while querying number of platforms");
      return PI_ERROR_INVALID_VALUE;
    }
    return PI_SUCCESS;
  }

  if (DeviceCount == 0) {
    /// No GPU entry to fill 'Devices' array
    return PI_SUCCESS;
  }

  if (Devices) {
    *Devices = Platform->PiDeviceCache.get();
  }
  return PI_SUCCESS;
}

// Check the device cache and load it if necessary.
pi_result _pi_platform::populateDeviceCacheIfNeeded() {
  std::lock_guard<std::mutex> Lock(PiDeviceCacheMutex);

  if (DeviceCachePopulated) {
    return PI_SUCCESS;
  }
  cm_support::CmDevice *CmDevice = nullptr;
  // TODO FIXME Implement proper version checking and reporting:
  // - version passed to cm_support::CreateCmDevice
  // - CmEmuVersion
  // - PluginVersion
  // - ESIMDEmuPluginOpaqueData::version
  //
  // PI_DEVICE_INFO_DRIVER_VERSION could report the ESIMDDeviceInterface
  // version, PI_PLATFORM_INFO_VERSION - the underlying libCM library version.
  unsigned int Version = 0;

  int Result = cm_support::CreateCmDevice(CmDevice, Version);

  if (Result != cm_support::CM_SUCCESS) {
    return PI_ERROR_INVALID_DEVICE;
  }

  // CM Device version info consists of two decimal numbers - major
  // and minor. Minor is single-digit. Version info is encoded into a
  // unsigned integer value = 100 * major + minor. Second from right
  // digit in decimal must be zero as it is used as 'dot'
  // REF - $CM_EMU/common/cm_version_defs.h - 'CURRENT_CM_VERSION'
  // e.g. CM version 7.3 => Device version = 703

  if (((Version / 10) % 10) != 0) {
    PiTrace("Invalid Arguments for piPlatformsGet of "
            "esimd_emulator (Platforms!=nullptr) "
            "while querying number of platforms");
    return PI_ERROR_INVALID_DEVICE;
  }

  std::ostringstream StrFormat;
  StrFormat << (int)(Version / 100) << "." << (int)(Version % 10);

  std::unique_ptr<_pi_device> Device(
      new _pi_device(this, CmDevice, StrFormat.str()));
  PiDeviceCache = std::move(Device);
  DeviceCachePopulated = true;
  return PI_SUCCESS;
}

pi_result piDeviceRetain(pi_device Device) {
  if (Device == nullptr) {
    return PI_ERROR_INVALID_DEVICE;
  }

  // CM supports only single device, which is root-device. 'Retain' is
  // No-op.
  return PI_SUCCESS;
}

pi_result piDeviceRelease(pi_device Device) {
  if (Device == nullptr) {
    return PI_ERROR_INVALID_DEVICE;
  }

  // CM supports only single device, which is root-device. 'Release'
  // is No-op.
  return PI_SUCCESS;
}

pi_result piDeviceGetInfo(pi_device Device, pi_device_info ParamName,
                          size_t ParamValueSize, void *ParamValue,
                          size_t *ParamValueSizeRet) {
  ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);

  switch (ParamName) {
  case PI_DEVICE_INFO_TYPE:
    return ReturnValue(PI_DEVICE_TYPE_GPU);
  case PI_DEVICE_INFO_PARENT_DEVICE:
    return ReturnValue(pi_device{0});
  case PI_DEVICE_INFO_PLATFORM:
    return ReturnValue(Device->Platform);
  case PI_DEVICE_INFO_NAME:
    return ReturnValue("ESIMD_EMULATOR");
  case PI_DEVICE_INFO_IMAGE_SUPPORT:
    return ReturnValue(pi_bool{true});
  case PI_DEVICE_INFO_DRIVER_VERSION:
    /// Combination of ESIMDEmuPluginDataVersion and
    /// ESIMDEmuPluginInterfaceVersion : 0.a.b
    /// a : ESIMDEmuPluginInterfaceVersion
    /// b : ESIMDEmuPluginDataVersion
    sprintf(ESimdEmuVersionString, "0.%d.%d", ESIMDEmuPluginInterfaceVersion,
            ESIMDEmuPluginDataVersion);
    return ReturnValue(ESimdEmuVersionString);
  case PI_DEVICE_INFO_VENDOR:
    return ReturnValue("Intel(R) Corporation");
  case PI_DEVICE_INFO_IMAGE2D_MAX_WIDTH:
    return ReturnValue(size_t{8192});
  case PI_DEVICE_INFO_IMAGE2D_MAX_HEIGHT:
    return ReturnValue(size_t{8192});
  case PI_DEVICE_INFO_HOST_UNIFIED_MEMORY:
    return ReturnValue(pi_bool{1});
  case PI_DEVICE_INFO_EXTENSIONS:
    // TODO : Populate return string accordingly - e.g. cl_khr_fp16,
    // cl_khr_fp64, cl_khr_int64_base_atomics,
    // cl_khr_int64_extended_atomics
    return ReturnValue("cl_khr_fp64");
  case PI_DEVICE_INFO_VERSION:
    return ReturnValue(Device->VersionStr.c_str());
  case PI_DEVICE_INFO_BUILD_ON_SUBDEVICE: // emulator doesn't support partition
    return ReturnValue(pi_bool{true});
  case PI_DEVICE_INFO_COMPILER_AVAILABLE:
    return ReturnValue(pi_bool{false});
  case PI_DEVICE_INFO_LINKER_AVAILABLE:
    return ReturnValue(pi_bool{false});
  case PI_DEVICE_INFO_MAX_COMPUTE_UNITS:
    return ReturnValue(pi_uint32{256});
  case PI_DEVICE_INFO_PARTITION_MAX_SUB_DEVICES:
    return ReturnValue(pi_uint32{0});
  case PI_DEVICE_INFO_PARTITION_PROPERTIES:
    return ReturnValue(pi_device_partition_property{0});
  case PI_DEVICE_INFO_VENDOR_ID:
    // '0x8086' : 'Intel HD graphics vendor ID'
    return ReturnValue(pi_uint32{0x8086});
  case PI_DEVICE_INFO_LOCAL_MEM_SIZE:
    // Default SLM_MAX_SIZE from CM_EMU
    return ReturnValue(pi_uint32{65536});
  case PI_DEVICE_INFO_MAX_WORK_GROUP_SIZE:
    return ReturnValue(size_t{256});
  case PI_DEVICE_INFO_MEM_BASE_ADDR_ALIGN:
    // Imported from level_zero
    return ReturnValue(pi_uint32{8});
  case PI_DEVICE_INFO_IMAGE3D_MAX_WIDTH:
  case PI_DEVICE_INFO_IMAGE3D_MAX_HEIGHT:
  case PI_DEVICE_INFO_IMAGE3D_MAX_DEPTH:
    // Default minimum values required by the SYCL specification.
    return ReturnValue(size_t{2048});
  case PI_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS:
    return ReturnValue(pi_uint32{3});
  case PI_DEVICE_INFO_PARTITION_TYPE:
    return ReturnValue(pi_device_partition_property{0});
  case PI_DEVICE_INFO_OPENCL_C_VERSION:
    return ReturnValue("");
  case PI_DEVICE_INFO_QUEUE_PROPERTIES:
    return ReturnValue(pi_queue_properties{PI_QUEUE_ON_DEVICE});
  case PI_DEVICE_INFO_MAX_WORK_ITEM_SIZES: {
    struct {
      size_t Arr[3];
    } MaxGroupSize = {{256, 256, 1}};
    return ReturnValue(MaxGroupSize);
  }
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_CHAR:
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_SHORT:
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_INT:
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_LONG:
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_FLOAT:
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_DOUBLE:
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_HALF:
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR:
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT:
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT:
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG:
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT:
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE:
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF:
    return ReturnValue(pi_uint32{1});

  // Imported from level_zero
  case PI_DEVICE_INFO_USM_HOST_SUPPORT:
  case PI_DEVICE_INFO_USM_DEVICE_SUPPORT:
  case PI_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT:
  case PI_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT:
  case PI_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT: {
    pi_uint64 Supported = 0;
    // TODO[1.0]: how to query for USM support now?
    if (true) {
      // TODO: Use ze_memory_access_capabilities_t
      Supported = PI_USM_ACCESS | PI_USM_ATOMIC_ACCESS |
                  PI_USM_CONCURRENT_ACCESS | PI_USM_CONCURRENT_ATOMIC_ACCESS;
    }
    return ReturnValue(Supported);
  }
  case PI_DEVICE_INFO_ADDRESS_BITS:
    return ReturnValue(
        pi_uint32{sizeof(void *) * std::numeric_limits<unsigned char>::digits});
  case PI_DEVICE_INFO_MAX_CLOCK_FREQUENCY:
    return ReturnValue(pi_uint32{1000});
  case PI_DEVICE_INFO_ENDIAN_LITTLE:
    return ReturnValue(pi_bool{true});
  case PI_DEVICE_INFO_AVAILABLE:
    return ReturnValue(pi_bool{true});
  case PI_DEVICE_INFO_MAX_READ_IMAGE_ARGS:
  case PI_DEVICE_INFO_MAX_WRITE_IMAGE_ARGS:
    /// TODO : Check
    return ReturnValue(pi_uint32{0});
  case PI_DEVICE_INFO_IMAGE_MAX_BUFFER_SIZE:
    /// TODO : Check. CM_MAX_1D_SURF_WIDTH from CM_EMU
    return ReturnValue(size_t{0x80000000});
  case PI_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE:
    /// TODO : Check
    return ReturnValue(size_t{0});
  case PI_DEVICE_INFO_MAX_SAMPLERS:
    /// TODO : Check. CM_MAX_SAMPLERS_PER_KERNEL from CM_EMU
    return ReturnValue(pi_uint32{16});
  case PI_DEVICE_INFO_MAX_PARAMETER_SIZE:
    /// TODO : Check
    return ReturnValue(size_t{32});
  case PI_DEVICE_INFO_HALF_FP_CONFIG:
  case PI_DEVICE_INFO_SINGLE_FP_CONFIG:
  case PI_DEVICE_INFO_DOUBLE_FP_CONFIG: {
    /// TODO : Check. half_type.hpp from CM_EMU
    uint64_t FPValue = PI_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT |
                       PI_FP_ROUND_TO_NEAREST | PI_FP_ROUND_TO_ZERO |
                       PI_FP_ROUND_TO_INF | PI_FP_INF_NAN | PI_FP_DENORM |
                       PI_FP_FMA;
    return ReturnValue(pi_uint64{FPValue});
  }
  case PI_DEVICE_INFO_GLOBAL_MEM_CACHE_TYPE:
    return ReturnValue(PI_DEVICE_MEM_CACHE_TYPE_READ_WRITE_CACHE);
  case PI_DEVICE_INFO_GLOBAL_MEM_CACHELINE_SIZE:
    // TODO : CHECK
    return ReturnValue(pi_uint32{64});
  case PI_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE:
    // TODO : CHECK
    return ReturnValue(pi_uint64{0});
  case PI_DEVICE_INFO_GLOBAL_MEM_SIZE:
    // TODO : CHECK
    return ReturnValue(pi_uint64{0});
  case PI_DEVICE_INFO_MAX_CONSTANT_BUFFER_SIZE:
    // TODO : CHECK
    return ReturnValue(pi_uint64{0});
  case PI_DEVICE_INFO_MAX_CONSTANT_ARGS:
    // TODO : CHECK
    return ReturnValue(pi_uint32{64});
  case PI_DEVICE_INFO_LOCAL_MEM_TYPE:
    // TODO : CHECK
    return ReturnValue(PI_DEVICE_LOCAL_MEM_TYPE_LOCAL);
  case PI_DEVICE_INFO_ERROR_CORRECTION_SUPPORT:
    return ReturnValue(pi_bool{false});
  case PI_DEVICE_INFO_PROFILING_TIMER_RESOLUTION:
    // TODO : CHECK
    return ReturnValue(size_t{0});
  case PI_DEVICE_INFO_BUILT_IN_KERNELS:
    // TODO : CHECK
    return ReturnValue("");
  case PI_DEVICE_INFO_PRINTF_BUFFER_SIZE:
    // TODO : CHECK
    return ReturnValue(size_t{1024});
  case PI_DEVICE_INFO_PREFERRED_INTEROP_USER_SYNC:
    return ReturnValue(pi_bool{false});
  case PI_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN:
    return ReturnValue(pi_device_affinity_domain{0});
  case PI_DEVICE_INFO_MAX_MEM_ALLOC_SIZE:
    // TODO : CHECK
    return ReturnValue(pi_uint64{0});
  case PI_DEVICE_INFO_EXECUTION_CAPABILITIES:
    // TODO : CHECK
    return ReturnValue(
        pi_device_exec_capabilities{PI_DEVICE_EXEC_CAPABILITIES_KERNEL});
  case PI_DEVICE_INFO_PROFILE:
    return ReturnValue("FULL_PROFILE");
  case PI_DEVICE_INFO_REFERENCE_COUNT:
    // TODO : CHECK
    return ReturnValue(pi_uint32{0});
  case PI_DEVICE_INFO_SUB_GROUP_SIZES_INTEL:
    return ReturnValue(size_t{1});

    CASE_PI_UNSUPPORTED(PI_DEVICE_INFO_MAX_NUM_SUB_GROUPS)
    CASE_PI_UNSUPPORTED(PI_DEVICE_INFO_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS)
    CASE_PI_UNSUPPORTED(PI_DEVICE_INFO_IL_VERSION)

    // Intel-specific extensions
    CASE_PI_UNSUPPORTED(PI_DEVICE_INFO_DEVICE_ID)
    CASE_PI_UNSUPPORTED(PI_DEVICE_INFO_PCI_ADDRESS)
    CASE_PI_UNSUPPORTED(PI_DEVICE_INFO_GPU_EU_COUNT)
    CASE_PI_UNSUPPORTED(PI_DEVICE_INFO_GPU_EU_SIMD_WIDTH)
    CASE_PI_UNSUPPORTED(PI_DEVICE_INFO_GPU_SLICES)
    CASE_PI_UNSUPPORTED(PI_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE)
    CASE_PI_UNSUPPORTED(PI_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE)
    CASE_PI_UNSUPPORTED(PI_DEVICE_INFO_MAX_MEM_BANDWIDTH)
    CASE_PI_UNSUPPORTED(PI_DEVICE_INFO_IMAGE_SRGB)
    CASE_PI_UNSUPPORTED(PI_DEVICE_INFO_ATOMIC_64)
    CASE_PI_UNSUPPORTED(PI_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES)
    CASE_PI_UNSUPPORTED(PI_EXT_ONEAPI_DEVICE_INFO_MAX_GLOBAL_WORK_GROUPS)
    CASE_PI_UNSUPPORTED(PI_EXT_ONEAPI_DEVICE_INFO_MAX_WORK_GROUPS_1D)
    CASE_PI_UNSUPPORTED(PI_EXT_ONEAPI_DEVICE_INFO_MAX_WORK_GROUPS_2D)
    CASE_PI_UNSUPPORTED(PI_EXT_ONEAPI_DEVICE_INFO_MAX_WORK_GROUPS_3D)

  default:
    DIE_NO_IMPLEMENTATION;
  }
  return PI_SUCCESS;
}

pi_result piDevicePartition(pi_device, const pi_device_partition_property *,
                            pi_uint32, pi_device *, pi_uint32 *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextDeviceGetNativeHandle(pi_device, pi_native_handle *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextDeviceCreateWithNativeHandle(pi_native_handle, pi_platform,
                                            pi_device *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piContextCreate(const pi_context_properties *Properties,
                          pi_uint32 NumDevices, const pi_device *Devices,
                          void (*PFnNotify)(const char *ErrInfo,
                                            const void *PrivateInfo, size_t CB,
                                            void *UserData),
                          void *UserData, pi_context *RetContext) {
  ARG_UNUSED(Properties);
  ARG_UNUSED(PFnNotify);
  ARG_UNUSED(UserData);

  if (NumDevices != 1) {
    return PI_ERROR_INVALID_VALUE;
  }
  if (Devices == nullptr) {
    return PI_ERROR_INVALID_DEVICE;
  }
  if (RetContext == nullptr) {
    return PI_ERROR_INVALID_VALUE;
  }

  try {
    /// Single-root-device
    *RetContext = new _pi_context(Devices[0]);
  } catch (const std::bad_alloc &) {
    return PI_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }
  return PI_SUCCESS;
}

pi_result piContextGetInfo(pi_context, pi_context_info, size_t, void *,
                           size_t *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextContextSetExtendedDeleter(pi_context,
                                         pi_context_extended_deleter, void *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextContextGetNativeHandle(pi_context, pi_native_handle *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextContextCreateWithNativeHandle(pi_native_handle, pi_uint32,
                                             const pi_device *, bool,
                                             pi_context *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piContextRetain(pi_context Context) {
  if (Context == nullptr) {
    return PI_ERROR_INVALID_CONTEXT;
  }

  ++(Context->RefCount);

  return PI_SUCCESS;
}

pi_result piContextRelease(pi_context Context) {
  if (Context == nullptr || (Context->RefCount <= 0)) {
    return PI_ERROR_INVALID_CONTEXT;
  }

  if (--(Context->RefCount) == 0) {
    /// TODO : Encapsulating accesses (add/remove) for
    /// Addr2CmBufferSVM
    std::lock_guard<std::mutex> Lock(Context->Addr2CmBufferSVMLock);
    for (auto &Entry : Context->Addr2CmBufferSVM) {
      Context->Device->CmDevicePtr->DestroyBufferSVM(Entry.second);
    }
    delete Context;
  }

  return PI_SUCCESS;
}

bool _pi_context::checkSurfaceArgument(pi_mem_flags Flags, void *HostPtr) {
  if (Flags & (PI_MEM_FLAGS_HOST_PTR_USE | PI_MEM_FLAGS_HOST_PTR_COPY)) {
    if (HostPtr == nullptr) {
      PiTrace("HostPtr argument is required for "
              "PI_MEM_FLAGS_HOST_PTR_USE/COPY");
      return false;
    }
    // COPY and USE are mutually exclusive
    if ((Flags & (PI_MEM_FLAGS_HOST_PTR_USE | PI_MEM_FLAGS_HOST_PTR_COPY)) ==
        (PI_MEM_FLAGS_HOST_PTR_USE | PI_MEM_FLAGS_HOST_PTR_COPY)) {
      PiTrace("PI_MEM_FLAGS_HOST_PTR_USE and _COPY cannot be used together");
      return false;
    }
  }
  return true;
}

pi_result piQueueCreate(pi_context Context, pi_device Device,
                        pi_queue_properties Properties, pi_queue *Queue) {
  ARG_UNUSED(Device);

  if (Properties & PI_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
    // TODO : Support Out-of-order Queue
    *Queue = nullptr;
    return PI_ERROR_INVALID_QUEUE_PROPERTIES;
  }

  cm_support::CmQueue *CmQueue = nullptr;

  int Result = Context->Device->CmDevicePtr->CreateQueue(CmQueue);
  if (Result != cm_support::CM_SUCCESS) {
    return PI_ERROR_INVALID_CONTEXT;
  }

  try {
    *Queue = new _pi_queue(Context, CmQueue);
  } catch (const std::bad_alloc &) {
    return PI_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }

  return PI_SUCCESS;
}

pi_result piQueueGetInfo(pi_queue, pi_queue_info, size_t, void *, size_t *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piQueueRetain(pi_queue Queue) {
  if (Queue == nullptr) {
    return PI_ERROR_INVALID_QUEUE;
  }
  ++(Queue->RefCount);
  return PI_SUCCESS;
}

pi_result piQueueRelease(pi_queue Queue) {
  if ((Queue == nullptr) || (Queue->CmQueuePtr == nullptr)) {
    return PI_ERROR_INVALID_QUEUE;
  }

  if (--(Queue->RefCount) == 0) {
    // CM's 'DestoryQueue' is no-op
    // Queue->Context->Device->CmDevicePTr->DestroyQueue(Queue->CmQueuePtr);
    delete Queue;
  }

  return PI_SUCCESS;
}

pi_result piQueueFinish(pi_queue) {
  // No-op as enqueued commands with ESIMD_EMULATOR plugin are blocking
  // ones that do not return until their completion - kernel execution
  // and memory read.
  CONTINUE_NO_IMPLEMENTATION;
}

pi_result piQueueFlush(pi_queue) {
  // No-op as enqueued commands with ESIMD_EMULATOR plugin are blocking
  // ones that do not return until their completion - kernel execution
  // and memory read.
  CONTINUE_NO_IMPLEMENTATION;
}

pi_result piextQueueGetNativeHandle(pi_queue, pi_native_handle *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextQueueCreateWithNativeHandle(pi_native_handle, pi_context,
                                           pi_device, bool, pi_queue *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piMemBufferCreate(pi_context Context, pi_mem_flags Flags, size_t Size,
                            void *HostPtr, pi_mem *RetMem,
                            const pi_mem_properties *properties) {
  ARG_UNUSED(properties);

  if ((Flags & PI_MEM_FLAGS_ACCESS_RW) == 0) {
    PiTrace("Invalid memory attribute for piMemBufferCreate");
    return PI_ERROR_INVALID_OPERATION;
  }

  if (Context == nullptr) {
    return PI_ERROR_INVALID_CONTEXT;
  }
  if (RetMem == nullptr) {
    return PI_ERROR_INVALID_VALUE;
  }

  // Flag & HostPtr argument sanity check
  if (!Context->checkSurfaceArgument(Flags, HostPtr)) {
    return PI_ERROR_INVALID_OPERATION;
  }

  char *MapBasePtr = nullptr;
  cm_surface_ptr_t CmBuf;
  cm_support::SurfaceIndex *CmIndex = nullptr;
  int Status = cm_support::CM_FAILURE;

  if (Flags & PI_MEM_FLAGS_HOST_PTR_USE) {
    CmBuf.tag = cm_surface_ptr_t::TypeUserProvidedBuffer;
    Status = Context->Device->CmDevicePtr->CreateBufferUP(
        static_cast<unsigned int>(Size), HostPtr, CmBuf.UPBufPtr);
    CmBuf.UPBufPtr->GetIndex(CmIndex);
  } else {
    CmBuf.tag = cm_surface_ptr_t::TypeRegularBuffer;
    Status = Context->Device->CmDevicePtr->CreateBuffer(
        static_cast<unsigned int>(Size), CmBuf.RegularBufPtr);
    CmBuf.RegularBufPtr->GetIndex(CmIndex);

    if (Flags & PI_MEM_FLAGS_HOST_PTR_COPY) {
      CmBuf.RegularBufPtr->WriteSurface(
          reinterpret_cast<const unsigned char *>(HostPtr), nullptr,
          static_cast<unsigned int>(Size));
    }
  }

  if (Status != cm_support::CM_SUCCESS) {
    return PI_ERROR_INVALID_OPERATION;
  }

  MapBasePtr =
      pi_cast<char *>(cm_support::get_surface_base_addr(CmIndex->get_data()));

  try {
    *RetMem =
        new _pi_buffer(Context, MapBasePtr, CmBuf, CmIndex->get_data(), Size);
  } catch (const std::bad_alloc &) {
    return PI_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }

  std::lock_guard<std::mutex> Lock{*PiESimdSurfaceMapLock};
  if (PiESimdSurfaceMap->find((*RetMem)->SurfaceIndex) !=
      PiESimdSurfaceMap->end()) {
    PiTrace("Failure from CM-managed buffer creation");
    return PI_ERROR_INVALID_MEM_OBJECT;
  }

  (*PiESimdSurfaceMap)[(*RetMem)->SurfaceIndex] = *RetMem;

  return PI_SUCCESS;
}

pi_result piMemGetInfo(pi_mem, pi_mem_info, size_t, void *, size_t *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piMemRetain(pi_mem Mem) {
  if (Mem == nullptr) {
    return PI_ERROR_INVALID_MEM_OBJECT;
  }
  ++(Mem->RefCount);
  return PI_SUCCESS;
}

pi_result piMemRelease(pi_mem Mem) {
  if ((Mem == nullptr) || (Mem->RefCount == 0)) {
    return PI_ERROR_INVALID_MEM_OBJECT;
  }

  if (--(Mem->RefCount) == 0) {
    // Removing Surface-map entry
    std::lock_guard<std::mutex> Lock{*PiESimdSurfaceMapLock};
    auto MapEntryIt = PiESimdSurfaceMap->find(Mem->SurfaceIndex);
    if (MapEntryIt == PiESimdSurfaceMap->end()) {
      PiTrace("Failure from Buffer/Image deletion");
      return PI_ERROR_INVALID_MEM_OBJECT;
    }
    PiESimdSurfaceMap->erase(MapEntryIt);
    delete Mem;
  }
  return PI_SUCCESS;
}

_pi_mem::~_pi_mem() {
  int Status = cm_support::CM_FAILURE;

  cm_support::CmDevice *CmDevice = Context->Device->CmDevicePtr;

  if (SurfacePtr.tag == cm_surface_ptr_t::TypeUserProvidedBuffer) {
    Status = CmDevice->DestroyBufferUP(SurfacePtr.UPBufPtr);
  } else if (SurfacePtr.tag == cm_surface_ptr_t::TypeRegularBuffer) {
    Status = CmDevice->DestroySurface(SurfacePtr.RegularBufPtr);
  } else if (SurfacePtr.tag == cm_surface_ptr_t::TypeUserProvidedImage) {
    Status = CmDevice->DestroySurface2DUP(SurfacePtr.UPImgPtr);
  } else if (SurfacePtr.tag == cm_surface_ptr_t::TypeRegularImage) {
    Status = CmDevice->DestroySurface(SurfacePtr.RegularImgPtr);
  }

  sycl::detail::pi::assertion(Status == cm_support::CM_SUCCESS &&
                              "Surface Deletion Failure from CM_EMU");

  for (auto mapit = Mappings.begin(); mapit != Mappings.end();) {
    mapit = Mappings.erase(mapit);
  }
}

cm_support::CM_SURFACE_FORMAT
ConvertPiImageFormatToCmFormat(const pi_image_format *PiFormat) {
  using ULongPair = std::pair<unsigned long, unsigned long>;
  using FmtMap = std::map<ULongPair, cm_support::CM_SURFACE_FORMAT>;
  static const FmtMap pi2cm = {
      {{PI_IMAGE_CHANNEL_TYPE_UNORM_INT8, PI_IMAGE_CHANNEL_ORDER_RGBA},
       cm_support::CM_SURFACE_FORMAT_A8R8G8B8},

      {{PI_IMAGE_CHANNEL_TYPE_UNORM_INT8, PI_IMAGE_CHANNEL_ORDER_ARGB},
       cm_support::CM_SURFACE_FORMAT_A8R8G8B8},

      {{PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8, PI_IMAGE_CHANNEL_ORDER_RGBA},
       cm_support::CM_SURFACE_FORMAT_A8R8G8B8},

      {{PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32, PI_IMAGE_CHANNEL_ORDER_RGBA},
       cm_support::CM_SURFACE_FORMAT_R32G32B32A32F},
  };
  auto Result = pi2cm.find(
      {PiFormat->image_channel_data_type, PiFormat->image_channel_order});
  if (Result != pi2cm.end()) {
    return Result->second;
  }
  return cm_support::CM_SURFACE_FORMAT_UNKNOWN;
}

pi_result piMemImageCreate(pi_context Context, pi_mem_flags Flags,
                           const pi_image_format *ImageFormat,
                           const pi_image_desc *ImageDesc, void *HostPtr,
                           pi_mem *RetImage) {
  if ((Flags & PI_MEM_FLAGS_ACCESS_RW) == 0) {
    PiTrace("Invalid memory attribute for piMemImageCreate");
    return PI_ERROR_INVALID_OPERATION;
  }

  if (ImageFormat == nullptr || ImageDesc == nullptr)
    return PI_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR;

  switch (ImageDesc->image_type) {
  case PI_MEM_TYPE_IMAGE2D:
    break;

    CASE_PI_UNSUPPORTED(PI_MEM_TYPE_IMAGE3D)
    CASE_PI_UNSUPPORTED(PI_MEM_TYPE_IMAGE2D_ARRAY)
    CASE_PI_UNSUPPORTED(PI_MEM_TYPE_IMAGE1D)
    CASE_PI_UNSUPPORTED(PI_MEM_TYPE_IMAGE1D_ARRAY)
    CASE_PI_UNSUPPORTED(PI_MEM_TYPE_IMAGE1D_BUFFER)

  default:
    return PI_ERROR_INVALID_MEM_OBJECT;
  }

  auto BytesPerPixel = 4;
  switch (ImageFormat->image_channel_data_type) {
  case PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32:
    BytesPerPixel = 16;
    break;
  case PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8:
  case PI_IMAGE_CHANNEL_TYPE_UNORM_INT8:
    BytesPerPixel = 4;
    break;
    CASE_PI_UNSUPPORTED(PI_IMAGE_CHANNEL_TYPE_SNORM_INT8)
    CASE_PI_UNSUPPORTED(PI_IMAGE_CHANNEL_TYPE_SNORM_INT16)
    CASE_PI_UNSUPPORTED(PI_IMAGE_CHANNEL_TYPE_UNORM_INT16)
    CASE_PI_UNSUPPORTED(PI_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565)
    CASE_PI_UNSUPPORTED(PI_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555)
    CASE_PI_UNSUPPORTED(PI_IMAGE_CHANNEL_TYPE_UNORM_INT_101010)
    CASE_PI_UNSUPPORTED(PI_IMAGE_CHANNEL_TYPE_SIGNED_INT8)
    CASE_PI_UNSUPPORTED(PI_IMAGE_CHANNEL_TYPE_SIGNED_INT16)
    CASE_PI_UNSUPPORTED(PI_IMAGE_CHANNEL_TYPE_SIGNED_INT32)
    CASE_PI_UNSUPPORTED(PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16)
    CASE_PI_UNSUPPORTED(PI_IMAGE_CHANNEL_TYPE_HALF_FLOAT)
    CASE_PI_UNSUPPORTED(PI_IMAGE_CHANNEL_TYPE_FLOAT)
  default:
    return PI_ERROR_IMAGE_FORMAT_NOT_SUPPORTED;
  }

  // Flag & HostPtr argument sanity check
  if (!Context->checkSurfaceArgument(Flags, HostPtr)) {
    return PI_ERROR_INVALID_OPERATION;
  }

  cm_support::CM_SURFACE_FORMAT CmSurfFormat =
      ConvertPiImageFormatToCmFormat(ImageFormat);
  if (CmSurfFormat == cm_support::CM_SURFACE_FORMAT_UNKNOWN) {
    return PI_ERROR_IMAGE_FORMAT_NOT_SUPPORTED;
  }

  char *MapBasePtr = nullptr;
  cm_surface_ptr_t CmImg;
  cm_support::SurfaceIndex *CmIndex = nullptr;
  int Status = cm_support::CM_SUCCESS;

  if (Flags & PI_MEM_FLAGS_HOST_PTR_USE) {
    CmImg.tag = cm_surface_ptr_t::TypeUserProvidedImage;
    Status = Context->Device->CmDevicePtr->CreateSurface2DUP(
        static_cast<unsigned int>(ImageDesc->image_width),
        static_cast<unsigned int>(ImageDesc->image_height), CmSurfFormat,
        HostPtr, CmImg.UPImgPtr);
    CmImg.UPImgPtr->GetIndex(CmIndex);
  } else {
    CmImg.tag = cm_surface_ptr_t::TypeRegularImage;
    Status = Context->Device->CmDevicePtr->CreateSurface2D(
        static_cast<unsigned int>(ImageDesc->image_width),
        static_cast<unsigned int>(ImageDesc->image_height), CmSurfFormat,
        CmImg.RegularImgPtr);
    CmImg.RegularImgPtr->GetIndex(CmIndex);

    if (Flags & PI_MEM_FLAGS_HOST_PTR_COPY) {
      CmImg.RegularImgPtr->WriteSurface(
          reinterpret_cast<const unsigned char *>(HostPtr), nullptr,
          static_cast<unsigned int>(ImageDesc->image_width *
                                    ImageDesc->image_height * BytesPerPixel));
    }
  }

  if (Status != cm_support::CM_SUCCESS) {
    return PI_ERROR_INVALID_OPERATION;
  }

  MapBasePtr =
      pi_cast<char *>(cm_support::get_surface_base_addr(CmIndex->get_data()));

  try {
    *RetImage = new _pi_image(Context, MapBasePtr, CmImg, CmIndex->get_data(),
                              ImageDesc->image_width, ImageDesc->image_height,
                              BytesPerPixel);
  } catch (const std::bad_alloc &) {
    return PI_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }

  std::lock_guard<std::mutex> Lock{*PiESimdSurfaceMapLock};
  if (PiESimdSurfaceMap->find((*RetImage)->SurfaceIndex) !=
      PiESimdSurfaceMap->end()) {
    PiTrace("Failure from CM-managed image creation");
    return PI_ERROR_INVALID_VALUE;
  }

  (*PiESimdSurfaceMap)[(*RetImage)->SurfaceIndex] = *RetImage;

  return PI_SUCCESS;
}

pi_result piextMemGetNativeHandle(pi_mem, pi_native_handle *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextMemCreateWithNativeHandle(pi_native_handle, pi_context, bool,
                                         pi_mem *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piProgramCreate(pi_context, const void *, size_t, pi_program *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piProgramCreateWithBinary(pi_context, pi_uint32, const pi_device *,
                                    const size_t *, const unsigned char **,
                                    size_t, const pi_device_binary_property *,
                                    pi_int32 *, pi_program *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piclProgramCreateWithBinary(pi_context, pi_uint32, const pi_device *,
                                      const size_t *, const unsigned char **,
                                      pi_int32 *, pi_program *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piclProgramCreateWithSource(pi_context, pi_uint32, const char **,
                                      const size_t *, pi_program *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piProgramGetInfo(pi_program, pi_program_info, size_t, void *,
                           size_t *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piProgramLink(pi_context, pi_uint32, const pi_device *, const char *,
                        pi_uint32, const pi_program *,
                        void (*)(pi_program, void *), void *, pi_program *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piProgramCompile(pi_program, pi_uint32, const pi_device *,
                           const char *, pi_uint32, const pi_program *,
                           const char **, void (*)(pi_program, void *),
                           void *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piProgramBuild(pi_program, pi_uint32, const pi_device *, const char *,
                         void (*)(pi_program, void *), void *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piProgramGetBuildInfo(pi_program, pi_device, pi_program_build_info,
                                size_t, void *, size_t *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piProgramRetain(pi_program) { DIE_NO_IMPLEMENTATION; }

pi_result piProgramRelease(pi_program) { DIE_NO_IMPLEMENTATION; }

pi_result piextProgramGetNativeHandle(pi_program, pi_native_handle *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextProgramCreateWithNativeHandle(pi_native_handle, pi_context, bool,
                                             pi_program *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piKernelCreate(pi_program, const char *, pi_kernel *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piKernelSetArg(pi_kernel, pi_uint32, size_t, const void *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextKernelSetArgMemObj(pi_kernel, pi_uint32, const pi_mem *) {
  DIE_NO_IMPLEMENTATION;
}

// Special version of piKernelSetArg to accept pi_sampler.
pi_result piextKernelSetArgSampler(pi_kernel, pi_uint32, const pi_sampler *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piKernelGetInfo(pi_kernel, pi_kernel_info, size_t, void *, size_t *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piKernelGetGroupInfo(pi_kernel, pi_device, pi_kernel_group_info,
                               size_t, void *, size_t *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piKernelGetSubGroupInfo(pi_kernel, pi_device,
                                  pi_kernel_sub_group_info, size_t,
                                  const void *, size_t, void *, size_t *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piKernelRetain(pi_kernel) { DIE_NO_IMPLEMENTATION; }

pi_result piKernelRelease(pi_kernel) { DIE_NO_IMPLEMENTATION; }

pi_result piEventCreate(pi_context, pi_event *) { DIE_NO_IMPLEMENTATION; }

pi_result piEventGetInfo(pi_event Event, pi_event_info ParamName,
                         size_t ParamValueSize, void *ParamValue,
                         size_t *ParamValueSizeRet) {
  if (ParamName != PI_EVENT_INFO_COMMAND_EXECUTION_STATUS) {
    DIE_NO_IMPLEMENTATION;
  }

  auto CheckAndFillStatus = [&](const cm_support::CM_STATUS &State) {
    pi_int32 Result = PI_EVENT_RUNNING;
    if (State == cm_support::CM_STATUS_FINISHED)
      Result = PI_EVENT_COMPLETE;
    if (ParamValue) {
      if (ParamValueSize < sizeof(Result))
        return PI_ERROR_INVALID_VALUE;
      *static_cast<pi_int32 *>(ParamValue) = Result;
    }
    if (ParamValueSizeRet) {
      *ParamValueSizeRet = sizeof(Result);
    }
    return PI_SUCCESS;
  };
  // Dummy event is already completed ones done by CM.
  if (Event->IsDummyEvent)
    return CheckAndFillStatus(cm_support::CM_STATUS_FINISHED);

  if (Event->CmEventPtr == nullptr)
    return PI_ERROR_INVALID_EVENT;

  cm_support::CM_STATUS Status;
  int32_t Result = Event->CmEventPtr->GetStatus(Status);
  if (Result != cm_support::CM_SUCCESS)
    return PI_ERROR_COMMAND_EXECUTION_FAILURE;

  return CheckAndFillStatus(Status);
}

pi_result piEventGetProfilingInfo(pi_event Event, pi_profiling_info ParamName,
                                  size_t ParamValueSize, void *ParamValue,
                                  size_t *ParamValueSizeRet) {
  ARG_UNUSED(Event);
  ARG_UNUSED(ParamName);
  ARG_UNUSED(ParamValueSize);
  ARG_UNUSED(ParamValue);
  ARG_UNUSED(ParamValueSizeRet);

  PiTrace("Warning : Profiling Not supported under PI_ESIMD_EMULATOR");
  return PI_SUCCESS;
}

pi_result piEventsWait(pi_uint32 NumEvents, const pi_event *EventList) {
  for (int i = 0; i < (int)NumEvents; i++) {
    if (EventList[i]->IsDummyEvent) {
      // Dummy event is already completed ones done by CM. Skip
      // waiting.
      continue;
    }
    if (EventList[i]->CmEventPtr == nullptr) {
      return PI_ERROR_INVALID_EVENT;
    }
    int Result = EventList[i]->CmEventPtr->WaitForTaskFinished();
    if (Result != cm_support::CM_SUCCESS) {
      return PI_ERROR_OUT_OF_RESOURCES;
    }
  }
  return PI_SUCCESS;
}

pi_result piEventSetCallback(pi_event, pi_int32,
                             void (*)(pi_event, pi_int32, void *), void *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piEventSetStatus(pi_event, pi_int32) { DIE_NO_IMPLEMENTATION; }

pi_result piEventRetain(pi_event Event) {
  if (Event == nullptr) {
    return PI_ERROR_INVALID_EVENT;
  }

  ++(Event->RefCount);

  return PI_SUCCESS;
}

pi_result piEventRelease(pi_event Event) {
  if (Event == nullptr || (Event->RefCount <= 0)) {
    return PI_ERROR_INVALID_EVENT;
  }

  if (--(Event->RefCount) == 0) {
    if (!Event->IsDummyEvent) {
      if ((Event->CmEventPtr == nullptr) || (Event->OwnerQueue == nullptr)) {
        return PI_ERROR_INVALID_EVENT;
      }
      int Result = Event->OwnerQueue->DestroyEvent(Event->CmEventPtr);
      if (Result != cm_support::CM_SUCCESS) {
        return PI_ERROR_INVALID_EVENT;
      }
    }
    delete Event;
  }

  return PI_SUCCESS;
}

pi_result piextEventGetNativeHandle(pi_event, pi_native_handle *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextEventCreateWithNativeHandle(pi_native_handle, pi_context, bool,
                                           pi_event *) {
  DIE_NO_IMPLEMENTATION;
}
pi_result piSamplerCreate(pi_context, const pi_sampler_properties *,
                          pi_sampler *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piSamplerGetInfo(pi_sampler, pi_sampler_info, size_t, void *,
                           size_t *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piSamplerRetain(pi_sampler) { DIE_NO_IMPLEMENTATION; }

pi_result piSamplerRelease(pi_sampler) { DIE_NO_IMPLEMENTATION; }

pi_result piEnqueueEventsWait(pi_queue, pi_uint32, const pi_event *,
                              pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piEnqueueEventsWaitWithBarrier(pi_queue, pi_uint32, const pi_event *,
                                         pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piEnqueueMemBufferRead(pi_queue Queue, pi_mem Src,
                                 pi_bool BlockingRead, size_t Offset,
                                 size_t Size, void *Dst,
                                 pi_uint32 NumEventsInWaitList,
                                 const pi_event *EventWaitList,
                                 pi_event *Event) {
  ARG_UNUSED(Queue);
  ARG_UNUSED(EventWaitList);

  /// TODO : Support Blocked read, 'Queue' handling
  if (BlockingRead) {
    PiTrace(
        "ESIMD_EMULATOR support for blocking piEnqueueMemBufferRead is NYI");
    return PI_ERROR_INVALID_OPERATION;
  }

  if (Offset != 0) {
    PiTrace("ESIMD_EMULATOR does not support buffer reading with offsets");
    return PI_ERROR_INVALID_ARG_VALUE;
  }

  if (NumEventsInWaitList != 0) {
    return PI_ERROR_INVALID_EVENT_WAIT_LIST;
  }

  _pi_buffer *buf = static_cast<_pi_buffer *>(Src);

  std::unique_ptr<_pi_event> RetEv{nullptr};
  if (Event) {
    RetEv = std::unique_ptr<_pi_event>(new _pi_event());
    RetEv->IsDummyEvent = true;
  }

  if (buf->SurfacePtr.tag == cm_surface_ptr_t::TypeUserProvidedBuffer) {
    // CM does not provide 'ReadSurface' call for 'User-Provided'
    // Surface. memcpy is used for BufferRead PI_API call.
    memcpy(Dst, buf->MapHostPtr, Size);
  } else {
    if (buf->SurfacePtr.tag != cm_surface_ptr_t::TypeRegularBuffer) {
      return PI_ERROR_INVALID_MEM_OBJECT;
    }
    int Status = buf->SurfacePtr.RegularBufPtr->ReadSurface(
        reinterpret_cast<unsigned char *>(Dst),
        nullptr, // event
        static_cast<uint64_t>(Size));

    if (Status != cm_support::CM_SUCCESS) {
      return PI_ERROR_INVALID_MEM_OBJECT;
    }
  }

  if (Event) {
    *Event = RetEv.release();
  }

  return PI_SUCCESS;
}

pi_result piEnqueueMemBufferReadRect(pi_queue, pi_mem, pi_bool,
                                     pi_buff_rect_offset, pi_buff_rect_offset,
                                     pi_buff_rect_region, size_t, size_t,
                                     size_t, size_t, void *, pi_uint32,
                                     const pi_event *, pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piEnqueueMemBufferWrite(pi_queue, pi_mem, pi_bool, size_t, size_t,
                                  const void *, pi_uint32, const pi_event *,
                                  pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piEnqueueMemBufferWriteRect(pi_queue, pi_mem, pi_bool,
                                      pi_buff_rect_offset, pi_buff_rect_offset,
                                      pi_buff_rect_region, size_t, size_t,
                                      size_t, size_t, const void *, pi_uint32,
                                      const pi_event *, pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piEnqueueMemBufferCopy(pi_queue, pi_mem, pi_mem, size_t, size_t,
                                 size_t, pi_uint32, const pi_event *,
                                 pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piEnqueueMemBufferCopyRect(pi_queue, pi_mem, pi_mem,
                                     pi_buff_rect_offset, pi_buff_rect_offset,
                                     pi_buff_rect_region, size_t, size_t,
                                     size_t, size_t, pi_uint32,
                                     const pi_event *, pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piEnqueueMemBufferFill(pi_queue, pi_mem, const void *, size_t, size_t,
                                 size_t, pi_uint32, const pi_event *,
                                 pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piEnqueueMemBufferMap(pi_queue Queue, pi_mem MemObj,
                                pi_bool BlockingMap, pi_map_flags MapFlags,
                                size_t Offset, size_t Size,
                                pi_uint32 NumEventsInWaitList,
                                const pi_event *EventWaitList, pi_event *Event,
                                void **RetMap) {
  ARG_UNUSED(Queue);
  ARG_UNUSED(BlockingMap);
  ARG_UNUSED(MapFlags);
  ARG_UNUSED(NumEventsInWaitList);
  ARG_UNUSED(EventWaitList);

  std::unique_ptr<_pi_event> RetEv{nullptr};
  pi_result ret = PI_SUCCESS;

  if (Event) {
    RetEv = std::unique_ptr<_pi_event>(new _pi_event());
    RetEv->IsDummyEvent = true;
  }

  // Real mapping does not occur here and CPU-accessible address is
  // returned as the actual memory space for the buffer is located in
  // CPU memory and the plug-in know its base address
  // ('_pi_mem::MapHostPtr')
  *RetMap = MemObj->MapHostPtr + Offset;

  {
    std::lock_guard<std::mutex> Lock{MemObj->MappingsMutex};
    auto Res = MemObj->Mappings.insert({*RetMap, {Offset, Size}});
    // False as the second value in pair means that mapping was not inserted
    // because mapping already exists.
    if (!Res.second) {
      ret = PI_ERROR_INVALID_VALUE;
      PiTrace("piEnqueueMemBufferMap: duplicate mapping detected");
    }
  }

  if (Event) {
    *Event = RetEv.release();
  }
  return ret;
}

pi_result piEnqueueMemUnmap(pi_queue Queue, pi_mem MemObj, void *MappedPtr,
                            pi_uint32 NumEventsInWaitList,
                            const pi_event *EventWaitList, pi_event *Event) {
  ARG_UNUSED(Queue);
  ARG_UNUSED(NumEventsInWaitList);
  ARG_UNUSED(EventWaitList);

  std::unique_ptr<_pi_event> RetEv{nullptr};
  pi_result ret = PI_SUCCESS;

  if (Event) {
    RetEv = std::unique_ptr<_pi_event>(new _pi_event());
    RetEv->IsDummyEvent = true;
  }

  // Real unmapping does not occur here and CPU-accessible address is
  // returned as the actual memory space for the buffer is located in
  // CPU memory and the plug-in knows its base address
  // ('_pi_mem::MapHostPtr')
  {
    std::lock_guard<std::mutex> Lock(MemObj->MappingsMutex);
    auto It = MemObj->Mappings.find(MappedPtr);
    if (It == MemObj->Mappings.end()) {
      ret = PI_ERROR_INVALID_VALUE;
      PiTrace("piEnqueueMemUnmap: unknown memory mapping");
    }
    MemObj->Mappings.erase(It);
  }

  if (Event) {
    *Event = RetEv.release();
  }

  return ret;
}

pi_result piMemImageGetInfo(pi_mem, pi_image_info, size_t, void *, size_t *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piEnqueueMemImageRead(pi_queue CommandQueue, pi_mem Image,
                                pi_bool BlockingRead, pi_image_offset Origin,
                                pi_image_region Region, size_t RowPitch,
                                size_t SlicePitch, void *Ptr,
                                pi_uint32 NumEventsInWaitList,
                                const pi_event *EventWaitList,
                                pi_event *Event) {
  ARG_UNUSED(CommandQueue);
  ARG_UNUSED(NumEventsInWaitList);
  ARG_UNUSED(EventWaitList);

  /// TODO : Support Blocked read, 'Queue' handling
  if (BlockingRead) {
    PiTrace("ESIMD_EMULATOR support for blocking piEnqueueMemImageRead is NYI");
    return PI_ERROR_INVALID_OPERATION;
  }

  // SlicePitch is for 3D image while ESIMD_EMULATOR does not
  // support. For 2D surfaces, SlicePitch must be 0.
  if (SlicePitch != 0) {
    PiTrace("ESIMD_EMULATOR does not support 3D-image");
    return PI_ERROR_INVALID_ARG_VALUE;
  }

  // CM_EMU does not support ReadSurface with offset
  if (Origin->x != 0 || Origin->y != 0 || Origin->z != 0) {
    PiTrace("ESIMD_EMULATOR does not support 2D-image reading with offsets");
    return PI_ERROR_INVALID_ARG_VALUE;
  }

  _pi_image *PiImg = static_cast<_pi_image *>(Image);

  std::unique_ptr<_pi_event> RetEv{nullptr};

  if (Event) {
    RetEv = std::unique_ptr<_pi_event>(new _pi_event());
    RetEv->IsDummyEvent = true;
  }

  size_t Size = RowPitch * (Region->height);
  if (PiImg->SurfacePtr.tag == cm_surface_ptr_t::TypeUserProvidedImage) {
    // CM does not provide 'ReadSurface' call for 'User-Provided'
    // Surface. memcpy is used for ImageRead PI_API call.
    memcpy(Ptr, PiImg->MapHostPtr, Size);
  } else {
    if (PiImg->SurfacePtr.tag != cm_surface_ptr_t::TypeRegularImage) {
      return PI_ERROR_INVALID_MEM_OBJECT;
    }
    int Status = PiImg->SurfacePtr.RegularImgPtr->ReadSurface(
        reinterpret_cast<unsigned char *>(Ptr),
        nullptr, // event
        static_cast<uint64_t>(Size));

    if (Status != cm_support::CM_SUCCESS) {
      return PI_ERROR_INVALID_MEM_OBJECT;
    }
  }

  if (Event) {
    *Event = RetEv.release();
  }

  return PI_SUCCESS;
}

pi_result piEnqueueMemImageWrite(pi_queue, pi_mem, pi_bool, pi_image_offset,
                                 pi_image_region, size_t, size_t, const void *,
                                 pi_uint32, const pi_event *, pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piEnqueueMemImageCopy(pi_queue, pi_mem, pi_mem, pi_image_offset,
                                pi_image_offset, pi_image_region, pi_uint32,
                                const pi_event *, pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piEnqueueMemImageFill(pi_queue, pi_mem, const void *, const size_t *,
                                const size_t *, pi_uint32, const pi_event *,
                                pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piMemBufferPartition(pi_mem, pi_mem_flags, pi_buffer_create_type,
                               void *, pi_mem *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result
piEnqueueKernelLaunch(pi_queue Queue, pi_kernel Kernel, pi_uint32 WorkDim,
                      const size_t *GlobalWorkOffset,
                      const size_t *GlobalWorkSize, const size_t *LocalWorkSize,
                      pi_uint32 NumEventsInWaitList,
                      const pi_event *EventWaitList, pi_event *Event) {
  ARG_UNUSED(Queue);
  ARG_UNUSED(NumEventsInWaitList);
  ARG_UNUSED(EventWaitList);

  const size_t LocalWorkSz[] = {1, 1, 1};

  if (Kernel == nullptr) {
    return PI_ERROR_INVALID_KERNEL;
  }

  if (WorkDim > 3 || WorkDim == 0) {
    return PI_ERROR_INVALID_WORK_GROUP_SIZE;
  }

  if (isNull(WorkDim, LocalWorkSize)) {
    LocalWorkSize = LocalWorkSz;
  }

  for (pi_uint32 I = 0; I < WorkDim; I++) {
    if ((GlobalWorkSize[I] % LocalWorkSize[I]) != 0) {
      return PI_ERROR_INVALID_WORK_GROUP_SIZE;
    }
  }

  std::unique_ptr<_pi_event> RetEv{nullptr};

  if (Event) {
    RetEv = std::unique_ptr<_pi_event>(new _pi_event());
    RetEv->IsDummyEvent = true;
  }

  switch (WorkDim) {
  case 1:
    InvokeImpl<1>::invoke(Kernel, GlobalWorkOffset, GlobalWorkSize,
                          LocalWorkSize);
    break;
  case 2:
    InvokeImpl<2>::invoke(Kernel, GlobalWorkOffset, GlobalWorkSize,
                          LocalWorkSize);
    break;
  case 3:
    InvokeImpl<3>::invoke(Kernel, GlobalWorkOffset, GlobalWorkSize,
                          LocalWorkSize);
    break;
  default:
    DIE_NO_IMPLEMENTATION;
    break;
  }

  if (Event) {
    *Event = RetEv.release();
  }

  return PI_SUCCESS;
}

pi_result piextKernelCreateWithNativeHandle(pi_native_handle, pi_context,
                                            pi_program, bool, pi_kernel *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextKernelGetNativeHandle(pi_kernel, pi_native_handle *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piEnqueueNativeKernel(pi_queue, void (*)(void *), void *, size_t,
                                pi_uint32, const pi_mem *, const void **,
                                pi_uint32, const pi_event *, pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextGetDeviceFunctionPointer(pi_device, pi_program, const char *,
                                        pi_uint64 *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextUSMHostAlloc(void **, pi_context, pi_usm_mem_properties *,
                            size_t, pi_uint32) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextUSMDeviceAlloc(void **, pi_context, pi_device,
                              pi_usm_mem_properties *, size_t, pi_uint32) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextUSMSharedAlloc(void **ResultPtr, pi_context Context,
                              pi_device Device,
                              pi_usm_mem_properties *Properties, size_t Size,
                              pi_uint32 Alignment) {
  ARG_UNUSED(Properties);
  ARG_UNUSED(Alignment);

  if (Context == nullptr || (Device != Context->Device)) {
    return PI_ERROR_INVALID_CONTEXT;
  }

  if (ResultPtr == nullptr) {
    return PI_ERROR_INVALID_OPERATION;
  }

  // 'Size' must be power of two in order to prevent memory corruption
  // error
  if ((Size & (Size - 1)) != 0) {
    Size = sycl::detail::getNextPowerOfTwo(Size);
  }

  cm_support::CmBufferSVM *Buf = nullptr;
  void *SystemMemPtr = nullptr;
  int32_t Result = Context->Device->CmDevicePtr->CreateBufferSVM(
      Size, SystemMemPtr, CM_SVM_ACCESS_FLAG_DEFAULT, Buf);

  if (Result != cm_support::CM_SUCCESS) {
    return PI_ERROR_OUT_OF_HOST_MEMORY;
  }
  *ResultPtr = SystemMemPtr;
  std::lock_guard<std::mutex> Lock(Context->Addr2CmBufferSVMLock);
  auto Iter = Context->Addr2CmBufferSVM.find(SystemMemPtr);
  if (Context->Addr2CmBufferSVM.end() != Iter) {
    return PI_ERROR_INVALID_MEM_OBJECT;
  }
  Context->Addr2CmBufferSVM[SystemMemPtr] = Buf;
  return PI_SUCCESS;
}

pi_result piextUSMFree(pi_context Context, void *Ptr) {
  if (Context == nullptr) {
    return PI_ERROR_INVALID_CONTEXT;
  }
  if (Ptr == nullptr) {
    return PI_ERROR_INVALID_OPERATION;
  }

  std::lock_guard<std::mutex> Lock(Context->Addr2CmBufferSVMLock);
  cm_support::CmBufferSVM *Buf = Context->Addr2CmBufferSVM[Ptr];
  if (Buf == nullptr) {
    return PI_ERROR_INVALID_MEM_OBJECT;
  }
  auto Count = Context->Addr2CmBufferSVM.erase(Ptr);
  if (Count != 1) {
    return PI_ERROR_INVALID_MEM_OBJECT;
  }
  int32_t Result = Context->Device->CmDevicePtr->DestroyBufferSVM(Buf);
  if (cm_support::CM_SUCCESS != Result) {
    return PI_ERROR_UNKNOWN;
  }
  return PI_SUCCESS;
}

pi_result piextKernelSetArgPointer(pi_kernel, pi_uint32, size_t, const void *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextUSMEnqueueMemset(pi_queue, void *, pi_int32, size_t, pi_uint32,
                                const pi_event *, pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextUSMEnqueueMemcpy(pi_queue, pi_bool, void *, const void *, size_t,
                                pi_uint32, const pi_event *, pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextUSMEnqueueMemAdvise(pi_queue, const void *, size_t,
                                   pi_mem_advice, pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextUSMGetMemAllocInfo(pi_context, const void *, pi_mem_alloc_info,
                                  size_t, void *, size_t *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piKernelSetExecInfo(pi_kernel, pi_kernel_exec_info, size_t,
                              const void *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextProgramSetSpecializationConstant(pi_program, pi_uint32, size_t,
                                                const void *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextDeviceSelectBinary(pi_device, pi_device_binary *,
                                  pi_uint32 RawImgSize, pi_uint32 *ImgInd) {
  /// TODO : Support multiple images and enable selection algorithm
  /// for the images
  if (RawImgSize != 1) {
    PiTrace("Only single device binary image is supported in ESIMD_EMULATOR");
    return PI_ERROR_INVALID_VALUE;
  }
  *ImgInd = 0;
  return PI_SUCCESS;
}

pi_result piextUSMEnqueuePrefetch(pi_queue, const void *, size_t,
                                  pi_usm_migration_flags, pi_uint32,
                                  const pi_event *, pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextPluginGetOpaqueData(void *, void **OpaqueDataReturn) {
  *OpaqueDataReturn = reinterpret_cast<void *>(PiESimdDeviceAccess);
  return PI_SUCCESS;
}

pi_result piTearDown(void *) {
  delete reinterpret_cast<sycl::detail::ESIMDEmuPluginOpaqueData *>(
      PiESimdDeviceAccess->data);
  delete PiESimdDeviceAccess;

  for (auto it = PiESimdSurfaceMap->begin(); it != PiESimdSurfaceMap->end();) {
    auto Mem = it->second;
    if (Mem != nullptr) {
      delete Mem;
    } // else { /* Null-entry for SLM_BTI */ }
    it = PiESimdSurfaceMap->erase(it);
  }
  return PI_SUCCESS;
}

const char SupportedVersion[] = _PI_ESIMD_PLUGIN_VERSION_STRING;

pi_result piPluginInit(pi_plugin *PluginInit) {
  if (PluginInit == nullptr) {
    return PI_ERROR_INVALID_VALUE;
  }

  // Check that the major version matches in PiVersion and SupportedVersion
  _PI_PLUGIN_VERSION_CHECK(PluginInit->PiVersion, SupportedVersion);

  size_t PluginVersionSize = sizeof(PluginInit->PluginVersion);
  if (strlen(_PI_H_VERSION_STRING) >= PluginVersionSize) {
    return PI_ERROR_INVALID_VALUE;
  }
  strncpy(PluginInit->PluginVersion, SupportedVersion, PluginVersionSize);

  PiESimdDeviceAccess = new sycl::detail::ESIMDEmuPluginOpaqueData();
  // 'version' to be compared with 'ESIMD_EMULATOR_DEVICE_REQUIRED_VER' defined
  // in device interface file
  PiESimdDeviceAccess->version = ESIMDEmuPluginDataVersion;
  PiESimdDeviceAccess->data =
      reinterpret_cast<void *>(new sycl::detail::ESIMDDeviceInterface());

  // Registering pre-defined surface index dedicated for SLM
  (*PiESimdSurfaceMap)[__ESIMD_DNS::SLM_BTI] = nullptr;

#define _PI_API(api)                                                           \
  (PluginInit->PiFunctionTable).api = (decltype(&::api))(&api);
#include <sycl/detail/pi.def>

  return PI_SUCCESS;
}

} // extern C
