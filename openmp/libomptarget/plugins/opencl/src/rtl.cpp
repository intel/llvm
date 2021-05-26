#if INTEL_COLLAB
//===--- Target RTLs Implementation ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is modified from https://github.com/daniel-schuermann/openmp.git.
// Thanks to Daniel Scheuermann, the author of rtl.cpp.
//
// RTL for SPIR-V/OpenCL machine
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <CL/cl.h>
#include <CL/cl_ext_intel.h>
#include <cassert>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdlib.h>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <Windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

#include "elf_light.h"
#include "omptargetplugin.h"
#include "omptarget-tools.h"
#include "rtl-trace.h"

#define TARGET_NAME OPENCL
#define DEBUG_PREFIX "Target " GETNAME(TARGET_NAME) " RTL"

/// Device type enumeration common to compiler and runtime
enum DeviceArch : uint64_t {
  DeviceArch_None   = 0,
  DeviceArch_Gen9   = 0x0001,
  DeviceArch_XeLP   = 0x0002,
  DeviceArch_XeHP   = 0x0004,
  DeviceArch_x86_64 = 0x0100
};

/// Mapping from device arch to GPU runtime's device identifiers
#ifdef _WIN32
/// For now, we need to depend on known published product names
std::map<uint64_t, std::vector<const char *>> DeviceArchMap {
  {
    DeviceArch_Gen9, {
      "HD Graphics",
      "UHD Graphics",
      "Pro Graphics",
      "Plus Graphics"
    }
  },
  {
    DeviceArch_XeLP, {
      "Xe MAX Graphics"
    }
  }
  // TODO: how to detect XeHP?
  // Using XeHP on Windows seems to be a rare case.
};
#else // !defined(_WIN32)
std::map<uint64_t, std::vector<uint32_t>> DeviceArchMap {
  {
    DeviceArch_Gen9, {
      0x0901, 0x0902, 0x0903, 0x0904, 0x1900, // SKL
      0x5900, // KBL
      0x3E00, 0x9B00, // CFL
    }
  },
  {
    DeviceArch_XeLP, {
      0xFF20, 0x9A00, // TGL
      0x4900, // DG1
      0x4C00, // RKL
      0x4600, // ADLS
    }
  }
};
#endif // !defined(_WIN32)

/// Interop support
namespace OCLInterop {
  // ID and names from openmp.org
  const int32_t Vendor = 8;
  const char *VendorName = GETNAME(intel);
  const int32_t FrId = 3;
  const char *FrName = GETNAME(opencl);

  // targetsync = -9, device_context = -8, ...,  fr_id = -1
  std::vector<const char *> IprNames {
    "targetsync",
    "device_context",
    "device",
    "platform",
    "device_num",
    "vendor_name",
    "vendor",
    "fr_name",
    "fr_id"
  };

  std::vector<const char *> IprTypeDescs {
    "cl_command_queue, opencl command queue handle",
    "cl_context, opencl context handle",
    "cl_device_id, opencl device handle",
    "cl_platform_id, opencl platform handle",
    "intptr_t, OpenMP device ID",
    "const char *, vendor name",
    "intptr_t, vendor ID",
    "const char *, foreign runtime name",
    "intptr_t, foreign runtime ID"
  };

  struct Property {
    // TODO
  };
}

int DebugLevel = 0;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// FIXME: we should actually include omp.h instead of declaring
//        these ourselves.
#if _WIN32
int __cdecl omp_get_max_teams(void);
int __cdecl omp_get_thread_limit(void);
double __cdecl omp_get_wtime(void);
int __cdecl __kmpc_global_thread_num(void *);
#else   // !_WIN32
int omp_get_max_teams(void) __attribute__((weak));
int omp_get_thread_limit(void) __attribute__((weak));
double omp_get_wtime(void) __attribute__((weak));
int __kmpc_global_thread_num(void *) __attribute__((weak));
#endif  // !_WIN32

#ifdef __cplusplus
}
#endif  // __cplusplus

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define OFFLOADSECTIONNAME "omp_offloading_entries"

//#pragma OPENCL EXTENSION cl_khr_spir : enable

/// Keep entries table per device.
struct FuncOrGblEntryTy {
  __tgt_target_table Table;
  std::vector<__tgt_offload_entry> Entries;
  std::vector<cl_kernel> Kernels;
  cl_program Program;
};

/// Loop descriptor
typedef struct {
  int64_t Lb;     // The lower bound of the i-th loop
  int64_t Ub;     // The upper bound of the i-th loop
  int64_t Stride; // The stride of the i-th loop
} TgtLoopDescTy;

typedef struct {
  int32_t NumLoops;        // Number of loops/dimensions
  int32_t DistributeDim;   // Dimensions lower than this one
                           // must end up in one WG
  TgtLoopDescTy Levels[3]; // Up to 3 loops
} TgtNDRangeDescTy;

/// Profile data
struct ProfileDataTy {
  struct TimingsTy {
    double host = 0.0;
    double device = 0.0;
  };

  std::map<std::string, TimingsTy> data;

  std::string alignLeft(size_t Width, std::string Str) {
    if (Str.size() < Width)
      return Str + std::string(Width - Str.size(), ' ');
    return Str;
  }

  void printData(int32_t deviceId, int32_t threadId, const char *deviceName,
                 int64_t resolution) {
    std::string profileSep(80, '=');
    std::string lineSep(80, '-');

    fprintf(stderr, "%s\n", profileSep.c_str());

    fprintf(stderr, "LIBOMPTARGET_PLUGIN_PROFILE(%s) for OMP DEVICE(%" PRId32
            ") %s, Thread %" PRId32 "\n", GETNAME(TARGET_NAME), deviceId,
            deviceName, threadId);

    fprintf(stderr, "%s\n", lineSep.c_str());

    const char *unit = resolution == 1000 ? "msec" : "usec";

    std::string kernelPrefix("Kernel ");
    size_t maxKeyLength = kernelPrefix.size() + 3;
    for (const auto &d : data)
      if (d.first.substr(0, kernelPrefix.size()) != kernelPrefix &&
          maxKeyLength < d.first.size())
        maxKeyLength = d.first.size();

    // Print kernel key and name
    int kernelId = 0;
    for (const auto &d: data) {
      if (d.first.substr(0, kernelPrefix.size()) == kernelPrefix)
        fprintf(stderr, "-- %s: %s\n",
                alignLeft(maxKeyLength, kernelPrefix +
                          std::to_string(kernelId++)).c_str(),
                d.first.substr(kernelPrefix.size()).c_str());
    }

    fprintf(stderr, "%s\n", lineSep.c_str());

    fprintf(stderr, "-- %s:     Host Time (%s)   Device Time (%s)\n",
            alignLeft(maxKeyLength, "Name").c_str(), unit, unit);

    double hostTotal = 0.0;
    double deviceTotal = 0.0;
    kernelId = 0;
    for (const auto &d : data) {
      double hostTime = 1e-9 * d.second.host * resolution;
      double deviceTime = 1e-9 * d.second.device * resolution;
      std::string key(d.first);

      if (d.first.substr(0, kernelPrefix.size()) == kernelPrefix)
        key = kernelPrefix + std::to_string(kernelId++);

      fprintf(stderr, "-- %s: %20.3f %20.3f\n",
              alignLeft(maxKeyLength, key).c_str(), hostTime, deviceTime);
      hostTotal += hostTime;
      deviceTotal += deviceTime;
    }
    fprintf(stderr, "-- %s: %20.3f %20.3f\n",
            alignLeft(maxKeyLength, "Total").c_str(), hostTotal, deviceTotal);
    fprintf(stderr, "%s\n", profileSep.c_str());
  }

  // for non-event profile
  void update(
      const char *name, cl_ulong host_elapsed, cl_ulong device_elapsed) {
    std::string key(name);
    TimingsTy &timings = data[key];
    timings.host += host_elapsed;
    timings.device += device_elapsed;
  }

  void update(
      const char *name, double host_elapsed, double device_elapsed) {
    std::string key(name);
    TimingsTy &timings = data[key];
    timings.host += host_elapsed;
    timings.device += device_elapsed;
  }

  // for event profile
  void update(const char *name, cl_event event) {
    cl_ulong host_begin = 0, host_end = 0;
    CALL_CLW_RET_VOID(clGetEventProfilingInfo, event,
        CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &host_begin, nullptr);
    CALL_CLW_RET_VOID(clGetEventProfilingInfo, event,
        CL_PROFILING_COMMAND_COMPLETE, sizeof(cl_ulong), &host_end, nullptr);
    cl_ulong device_begin = 0, device_end = 0;
    CALL_CLW_RET_VOID(clGetEventProfilingInfo, event,
        CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &device_begin, nullptr);
    CALL_CLW_RET_VOID(clGetEventProfilingInfo, event,
        CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &device_end, nullptr);
    update(name, host_end - host_begin, device_end - device_begin);
  }
}; // ProfileDataTy

// Platform-dependent information -- context and INTEL extension API
struct PlatformInfoTy {
  cl_platform_id Platform = nullptr;
  cl_context Context = nullptr;
  std::vector<const char *> ExtensionFunctionNames {
#define EXTENSION_FN_NAME(Fn) TO_STRING(Fn),
      FOR_EACH_EXTENSION_FN(EXTENSION_FN_NAME)
  };
  std::vector<void *> ExtensionFunctionPointers;

  PlatformInfoTy() = default;

  PlatformInfoTy(cl_platform_id platform, cl_context context) {
    Platform = platform;
    Context = context;
    ExtensionFunctionPointers.resize(ExtensionFunctionNames.size(), nullptr);
    for (int i = 0; i < ExtensionIdLast; i++) {
      CALL_CL_RV(ExtensionFunctionPointers[i],
                 clGetExtensionFunctionAddressForPlatform, platform,
                 ExtensionFunctionNames[i]);
      if (ExtensionFunctionPointers[i])
        IDP("Extension %s is found.\n", ExtensionFunctionNames[i]);
      else
        IDP("Warning: Extension %s is not found.\n", ExtensionFunctionNames[i]);
    }
  }
};

// OpenCL extensions status.
enum ExtensionStatusTy : uint8_t {
  // Default value.  It is unknown if the extension is supported.
  ExtensionStatusUnknown = 0,

  // Extension is disabled (either because it is unsupported or
  // due to user environment control).
  ExtensionStatusDisabled,

  // Extenstion is enabled.  An extension can only be used,
  // if it has this status after __tgt_rtl_load_binary.
  ExtensionStatusEnabled,
};

// A descriptor of OpenCL extensions with their statuses.
struct ExtensionsTy {
  ExtensionStatusTy UnifiedSharedMemory = ExtensionStatusUnknown;

  // Libdevice extensions that may be supported by device runtime.
  struct LibdeviceExtDescTy {
    const char *Name;
    const char *FallbackLibName;
    ExtensionStatusTy Status;
  };

  std::vector<LibdeviceExtDescTy> LibdeviceExtensions = {
#if ENABLE_LIBDEVICE_LINKING
    {
      "cl_intel_devicelib_cassert",
      "libomp-fallback-cassert.spv",
      ExtensionStatusUnknown
    },
    {
      "cl_intel_devicelib_math",
      "libomp-fallback-cmath.spv",
      ExtensionStatusUnknown
    },
    {
      "cl_intel_devicelib_math_fp64",
      "libomp-fallback-cmath-fp64.spv",
      ExtensionStatusUnknown
    },
    {
      "cl_intel_devicelib_complex",
      "libomp-fallback-complex.spv",
      ExtensionStatusUnknown
    },
    {
      "cl_intel_devicelib_complex_fp64",
      "libomp-fallback-complex-fp64.spv",
      ExtensionStatusUnknown
    },
#endif // ENABLE_LIBDEVICE_LINKING
  };

  // Initialize extensions' statuses for the given device.
  int32_t getExtensionsInfoForDevice(int32_t DeviceId);
};

/// Handler and argument for an asynchronous event.
/// Libomptarget is expected to provide this data.
struct AsyncEventTy {
  void (*handler)(void *); // Handler for the event
  void *arg;               // Argument to the handler
};

/// Data type used within this plugin for OCL event.
struct AsyncDataTy {
  AsyncEventTy *Event; // Data from Libomptarget
  // Add plugin data below
  int32_t DeviceId; // OMP device ID
  cl_mem MemToRelease; // Memory object to be released

  AsyncDataTy(AsyncEventTy *event, int32_t deviceId) : Event(event),
      DeviceId(deviceId), MemToRelease(nullptr) {}
};

/// Data transfer method
enum DataTransferMethodTy {
  DATA_TRANSFER_METHOD_INVALID = -1,   // Invalid
  DATA_TRANSFER_METHOD_CLMEM = 0,      // Use Buffer on SVM
  DATA_TRANSFER_METHOD_SVMMAP,         // Use SVMMap/Unmap
  DATA_TRANSFER_METHOD_SVMMEMCPY,      // Use SVMMemcpy
  DATA_TRANSFER_METHOD_LAST,
};

/// Buffer allocation information.
struct BufferInfoTy {
  void *Base;   // Base address
  int64_t Size; // Allocation size
};

/// Program data to be initialized.
/// TODO: include other runtime parameters if necessary.
struct ProgramData {
  int Initialized = 0;
  int NumDevices = 0;
  int DeviceNum = -1;
  uint32_t TotalEUs = 0;
  uint32_t HWThreadsPerEU = 0;
  uintptr_t DynamicMemoryLB = 0;
  uintptr_t DynamicMemoryUB = 0;
};

/// RTL flags
struct RTLFlagsTy {
  uint64_t CollectDataTransferLatency : 1;
  uint64_t EnableProfile : 1;
  uint64_t UseInteropQueueInorderAsync : 1;
  uint64_t UseInteropQueueInorderSharedSync : 1;
  uint64_t UseHostMemForUSM : 1;
  uint64_t UseDriverGroupSizes : 1;
  uint64_t EnableSimd : 1;
  uint64_t UseSVM : 1;
  uint64_t UseBuffer : 1;
  uint64_t UseSingleContext : 1;
  uint64_t UseImageOptions : 1;
  // Add new flags here
  uint64_t Reserved : 53;
  RTLFlagsTy() :
      CollectDataTransferLatency(0),
      EnableProfile(0),
      UseInteropQueueInorderAsync(0),
      UseInteropQueueInorderSharedSync(0),
      UseHostMemForUSM(0),
      UseDriverGroupSizes(0),
      EnableSimd(0),
      UseSVM(0),
      UseBuffer(0),
      UseSingleContext(0),
      UseImageOptions(1),
      Reserved(0) {}
};

/// Kernel properties.
struct KernelPropertiesTy {
  size_t Width = 0;
  size_t MaxThreadGroupSize = 0;
};

/// Specialization constants used for an OpenCL program compilation.
class SpecConstantsTy {
  std::vector<uint32_t> ConstantIds;
  std::vector<size_t> ConstantValueSizes;
  std::vector<const void *> ConstantValues;

public:
  SpecConstantsTy() = default;
  SpecConstantsTy(const SpecConstantsTy &) = delete;
  SpecConstantsTy(const SpecConstantsTy &&Other)
    : ConstantIds(std::move(Other.ConstantIds)),
      ConstantValueSizes(std::move(Other.ConstantValueSizes)),
      ConstantValues(std::move(Other.ConstantValues)) {}

  ~SpecConstantsTy() {
    for (auto I : ConstantValues) {
      const char *ValuePtr = reinterpret_cast<const char *>(I);
      delete[] ValuePtr;
    }
  }

  template <typename T>
  void addConstant(uint32_t Id, T Val) {
    const size_t ValSize = sizeof(Val);
    char *ValuePtr = new char[ValSize];
    *reinterpret_cast<T *>(ValuePtr) = Val;

    ConstantIds.push_back(Id);
    ConstantValueSizes.push_back(ValSize);
    ConstantValues.push_back(reinterpret_cast<void *>(ValuePtr));
  }

  void setProgramConstants(int32_t DeviceId, cl_program Program) const;
};

/// Class containing all the device information.
class RTLDeviceInfoTy {
  /// Type of the device version of the offload table.
  /// The type may not match the host offload table's type
  /// due to extensions.
  struct DeviceOffloadEntryTy {
    /// Common part with the host offload table.
    __tgt_offload_entry Base;
    /// Length of the Base.name string in bytes including
    /// the null terminator.
    size_t NameSize;
  };

  /// Looks up an external global variable with the given \p Name
  /// and \p Size in the device environment for device \p DeviceId.
  void *getVarDeviceAddr(int32_t DeviceId, const char *Name, size_t Size);
public:
  cl_uint numDevices;

  // Contains context and extension API
  std::map<cl_platform_id, PlatformInfoTy> PlatformInfos;

  // per device information
  std::vector<cl_platform_id> Platforms;
  std::vector<cl_context> Contexts;
  std::vector<cl_device_id> deviceIDs;
  // Internal device type ID
  std::vector<uint64_t> DeviceArchs;
  std::vector<int32_t> maxExecutionUnits;
  std::vector<size_t> maxWorkGroupSize;

  // A vector of descriptors of OpenCL extensions for each device.
  std::vector<ExtensionsTy> Extensions;
  std::vector<cl_command_queue> Queues;
  std::vector<cl_command_queue> QueuesInOrder;
  std::vector<FuncOrGblEntryTy> FuncGblEntries;
  std::vector<std::map<cl_kernel, KernelPropertiesTy>>
      KernelProperties;
  std::vector<std::map<void *, BufferInfoTy>> Buffers;
  std::vector<std::map<cl_kernel, std::set<void *>>> ImplicitArgs;
  std::vector<std::map<int32_t, ProfileDataTy>> Profiles;
  std::vector<std::vector<char>> Names;
  std::vector<bool> Initialized;
  std::vector<cl_ulong> SLMSize;
  std::vector<std::map<void *, int64_t>> DeviceAccessibleData;
  std::mutex *Mutexes;
  std::mutex *ProfileLocks;
  std::vector<std::vector<DeviceOffloadEntryTy>> OffloadTables;
  std::vector<std::set<void *>> ClMemBuffers;
  // Memory owned by the plugin
  std::vector<std::vector<void *>> OwnedMemory;

  // Requires flags
  int64_t RequiresFlags = OMP_REQ_UNDEFINED;

  RTLFlagsTy Flags;
  int32_t DataTransferLatency;
  int32_t DataTransferMethod;
  int64_t ProfileResolution;
  cl_device_type DeviceType;

  // OpenCL 2.0 builtins (like atomic_load_explicit and etc.) are used by
  // runtime, so we have to explicitly specify the "-cl-std=CL2.0" compilation
  // option. With it, the SPIR-V will be converted to LLVM IR with OpenCL 2.0
  // builtins. Otherwise, SPIR-V will be converted to LLVM IR with OpenCL 1.2
  // builtins.
  std::string CompilationOptions = "-cl-std=CL2.0 ";
  std::string UserCompilationOptions;
  std::string UserLinkingOptions;

  // Limit for the number of WIs in a WG.
  uint32_t ThreadLimit = 0;
  // Limit for the number of WGs.
  uint32_t NumTeams = 0;

  /// Dynamic kernel memory size
  size_t KernelDynamicMemorySize = 0; // Turned off by default

  // This is a factor applied to the number of WGs computed
  // for the execution, based on the HW characteristics.
  size_t SubscriptionRate = 1;

  // Spec constants used for all OpenCL programs.
  SpecConstantsTy CommonSpecConstants;

  RTLDeviceInfoTy() : numDevices(0), DataTransferLatency(0),
      DataTransferMethod(DATA_TRANSFER_METHOD_SVMMAP) {
    char *env;
    if ((env = readEnvVar("LIBOMPTARGET_DEBUG"))) {
      DebugLevel = std::stoi(env);
    }
    // set misc. flags

    // Get global OMP_THREAD_LIMIT for SPMD parallelization.
    int threadLimit = omp_get_thread_limit();
    IDP("omp_get_thread_limit() returned %" PRId32 "\n", threadLimit);
    // omp_get_thread_limit() would return INT_MAX by default.
    // NOTE: Windows.h defines max() macro, so we have to guard
    //       the call with parentheses.
    ThreadLimit = (threadLimit > 0 &&
        threadLimit != (std::numeric_limits<int32_t>::max)()) ?
        threadLimit : 0;

    // Global max number of teams.
    int numTeams = omp_get_max_teams();
    IDP("omp_get_max_teams() returned %" PRId32 "\n", numTeams);
    // omp_get_max_teams() would return INT_MAX by default.
    // NOTE: Windows.h defines max() macro, so we have to guard
    //       the call with parentheses.
    NumTeams = (numTeams > 0 &&
        numTeams != (std::numeric_limits<int32_t>::max)()) ?
        numTeams : 0;

    // Read LIBOMPTARGET_DATA_TRANSFER_LATENCY (experimental input)
    if ((env = readEnvVar("LIBOMPTARGET_DATA_TRANSFER_LATENCY"))) {
      std::string value(env);
      if (value.substr(0, 2) == "T,") {
        Flags.CollectDataTransferLatency = 1;
        int32_t usec = std::stoi(value.substr(2).c_str());
        DataTransferLatency = (usec > 0) ? usec : 0;
      }
    }

    // Read LIBOMPTARGET_DATA_TRANSFER_METHOD
    // Read LIBOMPTARGET_OPENCL_DATA_TRANSFER_METHOD
    if ((env = readEnvVar("LIBOMPTARGET_OPENCL_DATA_TRANSFER_METHOD",
                          "LIBOMPTARGET_DATA_TRANSFER_METHOD"))) {
      std::string value(env);
      DataTransferMethod = DATA_TRANSFER_METHOD_INVALID;
      if (value.size() == 1 && std::isdigit(value.c_str()[0])) {
        int method = std::stoi(env);
        if (method < DATA_TRANSFER_METHOD_LAST)
          DataTransferMethod = method;
      }
      if (DataTransferMethod == DATA_TRANSFER_METHOD_INVALID) {
        WARNING("Invalid data transfer method (%s) selected"
                " -- using default method.\n", env);
        DataTransferMethod = DATA_TRANSFER_METHOD_CLMEM;
      }
    }
    // Read LIBOMPTARGET_DEVICETYPE
    DeviceType = CL_DEVICE_TYPE_GPU;
    if ((env = readEnvVar("LIBOMPTARGET_DEVICETYPE"))) {
      std::string value(env);
      if (value == "GPU" || value == "gpu" || value == "")
        DeviceType = CL_DEVICE_TYPE_GPU;
      else if (value == "CPU" || value == "cpu")
        DeviceType = CL_DEVICE_TYPE_CPU;
      else
        WARNING("Invalid or unsupported LIBOMPTARGET_DEVICETYPE=%s\n", env);
    }
    IDP("Target device type is set to %s\n",
       (DeviceType == CL_DEVICE_TYPE_CPU) ? "CPU" : "GPU");

    if ((env = readEnvVar("LIBOMPTARGET_OPENCL_SUBSCRIPTION_RATE"))) {
      int32_t value = std::stoi(env);

      // Set some reasonable limits.
      if (value > 0 && value <= 0xFFFF)
        SubscriptionRate = value;
    }

    // Read LIBOMPTARGET_PROFILE
    ProfileResolution = 1000;
    if ((env = readEnvVar("LIBOMPTARGET_PLUGIN_PROFILE"))) {
      std::istringstream value(env);
      std::string token;
      while (std::getline(value, token, ',')) {
        if (token == "T" || token == "1")
          Flags.EnableProfile = 1;
        else if (token == "unit_usec" || token == "usec")
          ProfileResolution = 1000000;
      }
    }

    if ((env = readEnvVar("LIBOMPTARGET_ENABLE_SIMD"))) {
      std::string value(env);
      if (value == "T" || value == "1")
        Flags.EnableSimd = 1;
      else
        WARNING("Invalid or unsupported LIBOMPTARGET_ENABLE_SIMD=%s\n", env);
    }

    // Read LIBOMPTARGET_OPENCL_INTEROP_QUEUE
    // Two independent options can be specified as follows.
    // -- inorder_async: use a new in-order queue for asynchronous case
    //    (default: shared out-of-order queue)
    // -- inorder_shared_sync: use the existing shared in-order queue for
    //    synchronous case (default: new in-order queue).
    if ((env = readEnvVar("LIBOMPTARGET_OPENCL_INTEROP_QUEUE",
                          "LIBOMPTARGET_INTEROP_PIPE"))) {
      std::istringstream value(env);
      std::string token;
      while (std::getline(value, token, ',')) {
        if (token == "inorder_async") {
          Flags.UseInteropQueueInorderAsync = 1;
          IDP("    enabled in-order asynchronous separate queue\n");
        } else if (token == "inorder_shared_sync") {
          Flags.UseInteropQueueInorderSharedSync = 1;
          IDP("    enabled in-order synchronous shared queue\n");
        }
      }
    }

    if ((env = readEnvVar("LIBOMPTARGET_OPENCL_COMPILATION_OPTIONS"))) {
      UserCompilationOptions += env;
    }
    if ((env = readEnvVar("LIBOMPTARGET_OPENCL_LINKING_OPTIONS"))) {
      UserLinkingOptions += env;
    }

    // Read LIBOMPTARGET_USM_HOST_MEM
    if ((env = readEnvVar("LIBOMPTARGET_USM_HOST_MEM"))) {
      if (env[0] == 'T' || env[0] == 't' || env[0] == '1')
        Flags.UseHostMemForUSM = 1;
    }

    // Read LIBOMPTARGET_OPENCL_USE_SVM
    if ((env = readEnvVar("LIBOMPTARGET_OPENCL_USE_SVM"))) {
      if (env[0] == 'T' || env[0] == 't' || env[0] == '1')
        Flags.UseSVM = 1;
      else if (env[0] == 'F' || env[0] == 'f' || env[0] == '0')
        Flags.UseSVM = 0;
    }

    // Read LIBOMPTARGET_OPENCL_USE_BUFFER
    if ((env = readEnvVar("LIBOMPTARGET_OPENCL_USE_BUFFER"))) {
      if (env[0] == 'T' || env[0] == 't' || env[0] == '1')
        Flags.UseBuffer = 1;
    }

    // Read LIBOMPTARGET_USE_SINGLE_CONTEXT
    if ((env = readEnvVar("LIBOMPTARGET_USE_SINGLE_CONTEXT"))) {
      if (env[0] == 'T' || env[0] == 't' || env[0] == '1')
        Flags.UseSingleContext = 1;
    }

    // Read LIBOMPTARGET_DYNAMIC_MEMORY_SIZE=<SizeInMB>
    if ((env = readEnvVar("LIBOMPTARGET_DYNAMIC_MEMORY_SIZE"))) {
      size_t value = std::stoi(env);
      const size_t maxValue = 2048;
      if (value > maxValue) {
        IDP("Adjusted dynamic memory size to %zu MB\n", maxValue);
        value = maxValue;
      }
      KernelDynamicMemorySize = value << 20;
    }

    if ((env = readEnvVar("INTEL_LIBITTNOTIFY64"))) {
      // INTEL_LIBITTNOTIFY64 points to ittnotify_collector library.
      // Ignore empty path, but, otherwise, do not analyze it.
      if (env[0] != '\0') {
        CommonSpecConstants.addConstant<char>(0xFF747469, 1);
      }
    }

    if ((env = readEnvVar("LIBOMPTARGET_ONEAPI_USE_IMAGE_OPTIONS"))) {
      if (env[0] == 'T' || env[0] == 't' || env[0] == '1')
        Flags.UseImageOptions = 1;
      else if (env[0] == 'F' || env[0] == 'f' || env[0] == '0')
        Flags.UseImageOptions = 0;
    }
  }

  /// Read environment variable value with optional deprecated name
  char *readEnvVar(const char *Name, const char *OldName = nullptr) {
    if (!Name)
      return nullptr;
    char *value = std::getenv(Name);
    if (value || !OldName) {
      if (value)
        IDP("ENV: %s=%s\n", Name, value);
      return value;
    }
    value = std::getenv(OldName);
    if (value) {
      IDP("ENV: %s=%s\n", OldName, value);
      WARNING("%s is being deprecated. Use %s instead.\n", OldName, Name);
    }
    return value;
  }

  /// Return per-thread profile data
  ProfileDataTy &getProfiles(int32_t DeviceId) {
    int32_t gtid = __kmpc_global_thread_num(nullptr);
    ProfileLocks[DeviceId].lock();
    auto &profiles = Profiles[DeviceId];
    if (profiles.count(gtid) == 0)
      profiles.emplace(gtid, ProfileDataTy());
    auto &profileData = profiles[gtid];
    ProfileLocks[DeviceId].unlock();
    return profileData;
  }

  /// Return context for the given device ID
  cl_context getContext(int32_t DeviceId) {
    if (Flags.UseSingleContext)
      return PlatformInfos[Platforms[DeviceId]].Context;
    else
      return Contexts[DeviceId];
  }

  /// Return the extension function pointer for the given ID
  void *getExtensionFunctionPtr(int32_t DeviceId, int32_t ExtensionId) {
    auto platformId = Platforms[DeviceId];
    return PlatformInfos[platformId].ExtensionFunctionPointers[ExtensionId];
  }

  /// Return the extension function name for the given ID
  const char *getExtensionFunctionName(int32_t DeviceId, int32_t ExtensionId) {
    auto platformId = Platforms[DeviceId];
    return PlatformInfos[platformId].ExtensionFunctionNames[ExtensionId];
  }

  /// Check if extension function is available and enabled.
  bool isExtensionFunctionEnabled(int32_t DeviceId, int32_t ExtensionId) {
    if (!getExtensionFunctionPtr(DeviceId, ExtensionId))
      return false;

    switch (ExtensionId) {
    case clGetMemAllocInfoINTELId:
    case clHostMemAllocINTELId:
    case clDeviceMemAllocINTELId:
    case clSharedMemAllocINTELId:
    case clMemFreeINTELId:
    case clSetKernelArgMemPointerINTELId:
    case clEnqueueMemcpyINTELId:
      return Extensions[DeviceId].UnifiedSharedMemory == ExtensionStatusEnabled;
    default:
      return true;
    }
  }

  /// Loads the device version of the offload table for device \p DeviceId.
  /// The table is expected to have \p NumEntries entries.
  /// Returns true, if the load was successful, false - otherwise.
  bool loadOffloadTable(int32_t DeviceId, size_t NumEntries);

  /// Deallocates resources allocated for the device offload table
  /// of \p DeviceId device.
  void unloadOffloadTable(int32_t DeviceId);

  /// Looks up an OpenMP declare target global variable with the given
  /// \p Name and \p Size in the device environment for device \p DeviceId.
  /// The lookup is first done via the device offload table. If it fails,
  /// then the lookup falls back to non-OpenMP specific lookup on the device.
  void *getOffloadVarDeviceAddr(
      int32_t DeviceId, const char *Name, size_t Size);

  /// Initialize program data on device
  int32_t initProgramData(int32_t DeviceId);

  /// Allocate cl_mem data
  void *allocDataClMem(int32_t DeviceId, size_t Size);
};

#ifdef _WIN32
#define __ATTRIBUTE__(X)
#else
#define __ATTRIBUTE__(X)  __attribute__((X))
#endif // _WIN32

static RTLDeviceInfoTy *DeviceInfo = nullptr;

__ATTRIBUTE__(constructor(101)) void init() {
  IDP("Init OpenCL plugin!\n");
  DeviceInfo = new RTLDeviceInfoTy();
}

__ATTRIBUTE__(destructor(101)) void deinit() {
  IDP("Deinit OpenCL plugin!\n");
  delete DeviceInfo;
  DeviceInfo = nullptr;
}

#if _WIN32
extern "C" BOOL WINAPI
DllMain(HINSTANCE const instance, // handle to DLL module
        DWORD const reason,       // reason for calling function
        LPVOID const reserved)    // reserved
{
  // Perform actions based on the reason for calling.
  switch (reason) {
  case DLL_PROCESS_ATTACH:
    // Initialize once for each new process.
    // Return FALSE to fail DLL load.
    init();
    break;

  case DLL_THREAD_ATTACH:
    // Do thread-specific initialization.
    break;

  case DLL_THREAD_DETACH:
    // Do thread-specific cleanup.
    break;

  case DLL_PROCESS_DETACH:
    break;
  }
  return TRUE; // Successful DLL_PROCESS_ATTACH.
}
#endif // _WIN32

// Helper class to collect time intervals for host and device.
// The interval is managed by start()/stop() methods.
// The automatic flush of the time interval happens at the object's
// destruction.
class ProfileIntervalTy {
  // A timer interval may be either disabled, paused or running.
  // Interval may switch from paused to running and from running
  // to paused. Interval may switch to disabled start from any state,
  // and it cannot switch to disabled to anything else.
  enum TimerStatusTy {
    Disabled,
    Paused,
    Running
  };

  // Cumulative times collected by this interval so far.
  double DeviceElapsed = 0.0;
  double HostElapsed = 0.0;

  // Temporary timer values initialized at each interval
  // (re)start.
  cl_ulong DeviceTimeTemp = 0;
  cl_ulong HostTimeTemp = 0;

  // The interval name as seen in the profile data output.
  std::string Name;

  // OpenMP device id.
  int32_t DeviceId;

  // OpenCL device id.
  cl_device_id ClDeviceId;

  // Current status of the interval.
  TimerStatusTy Status;

public:
  // Create new timer interval for the given OpenMP device
  // and with the given name (which will be used for the profile
  // data output).
  ProfileIntervalTy(const char *Name, int32_t DeviceId)
    : Name(Name), DeviceId(DeviceId),
      ClDeviceId(DeviceInfo->deviceIDs[DeviceId]) {
    if (DeviceInfo->Flags.EnableProfile)
      // Start the interval paused.
      Status = TimerStatusTy::Paused;
    else
      // Disable the interval for good.
      Status = TimerStatusTy::Disabled;
  }

  // The destructor automatically updates the profile data.
  ~ProfileIntervalTy() {
    if (Status == TimerStatusTy::Disabled)
      return;
    if (Status == TimerStatusTy::Running) {
      Status = TimerStatusTy::Disabled;
      WARNING("profiling timer '%s' for OpenMP device (%" PRId32 ") %s "
              "is disabled due to start/stop mismatch.\n",
              Name.c_str(), DeviceId, DeviceInfo->Names[DeviceId].data());
      return;
    }

    DeviceInfo->getProfiles(DeviceId).update(
        Name.c_str(), HostElapsed, DeviceElapsed);
  }

  // Trigger interval start.
  void start() {
    if (Status == TimerStatusTy::Disabled)
      return;
    if (Status == TimerStatusTy::Running) {
      Status = TimerStatusTy::Disabled;
      WARNING("profiling timer '%s' for OpenMP device (%" PRId32 ") %s "
              "is disabled due to start/stop mismatch.\n",
              Name.c_str(), DeviceId, DeviceInfo->Names[DeviceId].data());
      return;
    }
    cl_int rc;
    CALL_CL(rc, clGetDeviceAndHostTimer, ClDeviceId, &DeviceTimeTemp,
            &HostTimeTemp);
    if (rc != CL_SUCCESS) {
      Status = TimerStatusTy::Disabled;
      WARNING("profiling timer '%s' for OpenMP device (%" PRId32 ") %s "
              "is disabled due to invalid OpenCL timer.\n",
              Name.c_str(), DeviceId, DeviceInfo->Names[DeviceId].data());
      return;
    }
    Status = TimerStatusTy::Running;
  }

  // Trigger interval stop (actually, a pause).
  void stop() {
    if (Status == TimerStatusTy::Disabled)
      return;
    if (Status == TimerStatusTy::Paused) {
      Status = TimerStatusTy::Disabled;
      WARNING("profiling timer '%s' for OpenMP device (%" PRId32 ") %s "
              "is disabled due to start/stop mismatch.\n",
              Name.c_str(), DeviceId, DeviceInfo->Names[DeviceId].data());
      return;
    }

    cl_ulong DeviceTime, HostTime;
    cl_int rc;
    CALL_CL(rc, clGetDeviceAndHostTimer, ClDeviceId, &DeviceTime, &HostTime);
    if (rc != CL_SUCCESS) {
      Status = TimerStatusTy::Disabled;
      WARNING("profiling timer '%s' for OpenMP device (%" PRId32 ") %s "
              "is disabled due to invalid OpenCL timer.\n",
              Name.c_str(), DeviceId, DeviceInfo->Names[DeviceId].data());
      return;
    }

    if (DeviceTime < DeviceTimeTemp || HostTime < HostTimeTemp) {
      Status = TimerStatusTy::Disabled;
      WARNING("profiling timer '%s' for OpenMP device (%" PRId32 ") %s "
              "is disabled due to timer overflow.\n",
              Name.c_str(), DeviceId, DeviceInfo->Names[DeviceId].data());
      return;
    }

    DeviceElapsed +=
        static_cast<double>(DeviceTime) - static_cast<double>(DeviceTimeTemp);
    HostElapsed +=
        static_cast<double>(HostTime) - static_cast<double>(HostTimeTemp);
    Status = TimerStatusTy::Paused;
  }
}; // ProfileIntervalTy

/// Clean-up routine to be registered by std::atexit().
static void closeRTL() {
  for (uint32_t i = 0; i < DeviceInfo->numDevices; i++) {
    if (!DeviceInfo->Initialized[i])
      continue;
    if (DeviceInfo->Flags.EnableProfile) {
      for (auto &profile : DeviceInfo->Profiles[i])
        profile.second.printData(i, profile.first, DeviceInfo->Names[i].data(),
                                 DeviceInfo->ProfileResolution);
    }
    if (OMPT_ENABLED) {
      OMPT_CALLBACK(ompt_callback_device_unload, i, 0 /* module ID */);
      OMPT_CALLBACK(ompt_callback_device_finalize, i);
    }
    // Making OpenCL calls during process exit on Windows is unsafe.
    for (auto kernel : DeviceInfo->FuncGblEntries[i].Kernels) {
      if (kernel)
        CALL_CL_EXIT_FAIL(clReleaseKernel, kernel);
    }
    // No entries may exist if offloading was done through MKL
    if (DeviceInfo->FuncGblEntries[i].Program)
      CALL_CL_EXIT_FAIL(clReleaseProgram, DeviceInfo->FuncGblEntries[i].Program);

    CALL_CL_EXIT_FAIL(clReleaseCommandQueue, DeviceInfo->Queues[i]);

    if (DeviceInfo->QueuesInOrder[i])
      CALL_CL_EXIT_FAIL(clReleaseCommandQueue, DeviceInfo->QueuesInOrder[i]);

    DeviceInfo->unloadOffloadTable(i);

    for (auto mem : DeviceInfo->OwnedMemory[i])
      CALL_CL_EXT_VOID(i, clMemFreeINTEL, DeviceInfo->getContext(i), mem);

    if (!DeviceInfo->Flags.UseSingleContext)
      CALL_CL_EXIT_FAIL(clReleaseContext, DeviceInfo->Contexts[i]);
  }

  if (DeviceInfo->Flags.UseSingleContext)
    for (auto platformInfo : DeviceInfo->PlatformInfos)
      CALL_CL_EXIT_FAIL(clReleaseContext, platformInfo.second.Context);

  delete[] DeviceInfo->Mutexes;
  delete[] DeviceInfo->ProfileLocks;
  IDP("Closed RTL successfully\n");
}

static std::string getDeviceRTLPath(const char *basename) {
  std::string rtl_path;
#ifdef _WIN32
  char path[_MAX_PATH];
  HMODULE module = nullptr;
  if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                          GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          (LPCSTR) &DeviceInfo, &module))
    return rtl_path;
  if (!GetModuleFileNameA(module, path, sizeof(path)))
    return rtl_path;
  rtl_path = path;
#else
  Dl_info rtl_info;
  if (!dladdr(&DeviceInfo, &rtl_info))
    return rtl_path;
  rtl_path = rtl_info.dli_fname;
#endif
  size_t split = rtl_path.find_last_of("/\\");
  rtl_path.replace(split + 1, std::string::npos, basename);
  return rtl_path;
}

/// Initialize program data.
/// TODO: consider moving allocation of static buffers in device RTL to here
///       as it requires device information.
int32_t RTLDeviceInfoTy::initProgramData(int32_t deviceId) {
  uint32_t totalEUs = maxExecutionUnits[deviceId];
  uint32_t numThreadsPerEU;

  switch (DeviceArchs[deviceId]) {
  case DeviceArch_Gen9:
  case DeviceArch_XeLP:
    numThreadsPerEU = 7;
    break;
  case DeviceArch_XeHP:
    numThreadsPerEU = 8;
    break;
  default:
    numThreadsPerEU = 1;
  }

  // Allocate dynamic memory for in-kernel allocation
  void *memLB = 0;
  uintptr_t memUB = 0;
  if (KernelDynamicMemorySize > 0) {
    cl_int rc;
    CALL_CL_EXT_RVRC(deviceId, memLB, clDeviceMemAllocINTEL, rc,
                     getContext(deviceId), deviceIDs[deviceId], nullptr,
                     KernelDynamicMemorySize, 0);
  }
  if (memLB) {
    OwnedMemory[deviceId].push_back(memLB);
    memUB = (uintptr_t)memLB + KernelDynamicMemorySize;
  }

  ProgramData hostData = {
    1,                   // Initialized
    (int32_t)numDevices, // Number of devices
    deviceId,            // Device ID
    totalEUs,            // Total EUs
    numThreadsPerEU,     // HW threads per EU
    (uintptr_t)memLB,    // Dynamic memory LB
    memUB                // Dynamic memory UB
  };

  return OFFLOAD_SUCCESS;
}

void *RTLDeviceInfoTy::getOffloadVarDeviceAddr(
    int32_t DeviceId, const char *Name, size_t Size) {
  IDP(
     "Looking up OpenMP global variable '%s' of size %zu bytes on device %d.\n",
     Name, Size, DeviceId);

  std::vector<DeviceOffloadEntryTy> &OffloadTable = OffloadTables[DeviceId];
  if (!OffloadTable.empty()) {
    size_t NameSize = strlen(Name) + 1;
    auto I = std::lower_bound(
        OffloadTable.begin(), OffloadTable.end(), Name,
        [NameSize](const DeviceOffloadEntryTy &E, const char *Name) {
          return strncmp(E.Base.name, Name, NameSize) < 0;
        });

    if (I != OffloadTable.end() &&
        strncmp(I->Base.name, Name, NameSize) == 0) {
      IDP("Global variable '%s' found in the offload table at position %zu.\n",
         Name, std::distance(OffloadTable.begin(), I));
      return I->Base.addr;
    }

    IDP("Warning: global variable '%s' was not found in the offload table.\n",
       Name);
  } else
    IDP("Warning: offload table is not loaded for device %d.\n", DeviceId);

  // Fallback to the lookup by name.
  return getVarDeviceAddr(DeviceId, Name, Size);
}

void *RTLDeviceInfoTy::getVarDeviceAddr(
    int32_t DeviceId, const char *Name, size_t Size) {
  size_t DeviceSize = 0;
  void *TgtAddr = nullptr;
  IDP(
     "Looking up device global variable '%s' of size %zu bytes on device %d.\n",
     Name, Size, DeviceId);
  // TODO: use device API to get variable address by name.

  if (DeviceSize == 0) {
    IDP("Warning: global variable lookup failed.\n");
    return nullptr;
  }

  IDP("Global variable lookup succeeded.\n");
  return TgtAddr;
}

bool RTLDeviceInfoTy::loadOffloadTable(int32_t DeviceId, size_t NumEntries) {
  const char *OffloadTableSizeVarName = "__omp_offloading_entries_table_size";
  void *OffloadTableSizeVarAddr =
      getVarDeviceAddr(DeviceId, OffloadTableSizeVarName, sizeof(int64_t));

  if (!OffloadTableSizeVarAddr) {
    IDP("Warning: cannot get device value for global variable '%s'.\n",
       OffloadTableSizeVarName);
    return false;
  }

  int64_t TableSizeVal = 0;
  CALL_CL_EXT_RET(DeviceId, false, clEnqueueMemcpyINTEL, Queues[DeviceId],
                  CL_TRUE, &TableSizeVal, OffloadTableSizeVarAddr,
                  sizeof(int64_t), 0, nullptr, nullptr);
  size_t TableSize = (size_t)TableSizeVal;

  if ((TableSize % sizeof(DeviceOffloadEntryTy)) != 0) {
    IDP("Warning: offload table size (%zu) is not a multiple of %zu.\n",
       TableSize, sizeof(DeviceOffloadEntryTy));
    return false;
  }

  size_t DeviceNumEntries = TableSize / sizeof(DeviceOffloadEntryTy);

  if (NumEntries != DeviceNumEntries) {
    IDP("Warning: number of entries in host and device "
       "offload tables mismatch (%zu != %zu).\n",
       NumEntries, DeviceNumEntries);
  }

  const char *OffloadTableVarName = "__omp_offloading_entries_table";
  void *OffloadTableVarAddr =
      getVarDeviceAddr(DeviceId, OffloadTableVarName, TableSize);
  if (!OffloadTableVarAddr) {
    IDP("Warning: cannot get device value for global variable '%s'.\n",
       OffloadTableVarName);
    return false;
  }

  OffloadTables[DeviceId].resize(DeviceNumEntries);
  CALL_CL_EXT_RET(DeviceId, false, clEnqueueMemcpyINTEL, Queues[DeviceId],
                  CL_TRUE, OffloadTables[DeviceId].data(), OffloadTableVarAddr,
                  TableSize, 0, nullptr, nullptr);
  std::vector<DeviceOffloadEntryTy> &DeviceTable = OffloadTables[DeviceId];

  size_t I = 0;
  const char *PreviousName = "";
  bool PreviousIsVar = false;

  for (; I < DeviceNumEntries; ++I) {
    DeviceOffloadEntryTy &Entry = DeviceTable[I];
    size_t NameSize = Entry.NameSize;
    void *NameTgtAddr = Entry.Base.name;
    Entry.Base.name = nullptr;

    if (NameSize == 0) {
      IDP("Warning: offload entry (%zu) with 0 size.\n", I);
      break;
    }
    if (NameTgtAddr == nullptr) {
      IDP("Warning: offload entry (%zu) with invalid name.\n", I);
      break;
    }

    Entry.Base.name = new char[NameSize];
    CALL_CL_EXT_RET(DeviceId, false, clEnqueueMemcpyINTEL, Queues[DeviceId],
                    CL_TRUE, Entry.Base.name, NameTgtAddr, NameSize, 0, nullptr,
                    nullptr);
    if (strnlen(Entry.Base.name, NameSize) != NameSize - 1) {
      IDP("Warning: offload entry's name has wrong size.\n");
      break;
    }

    int Cmp = strncmp(PreviousName, Entry.Base.name, NameSize);
    if (Cmp > 0) {
      IDP("Warning: offload table is not sorted.\n"
         "Warning: previous name is '%s'.\n"
         "Warning:  current name is '%s'.\n",
         PreviousName, Entry.Base.name);
      break;
    } else if (Cmp == 0 && (PreviousIsVar || Entry.Base.addr)) {
      // The names are equal. This should never happen for
      // offload variables, but we allow this for offload functions.
      IDP("Warning: duplicate names (%s) in offload table.\n", PreviousName);
      break;
    }
    PreviousName = Entry.Base.name;
    PreviousIsVar = (Entry.Base.addr != nullptr);
  }

  if (I != DeviceNumEntries) {
    // Errors during the table processing.
    // Deallocate all memory allocated in the loop.
    for (size_t J = 0; J <= I; ++J) {
      DeviceOffloadEntryTy &Entry = DeviceTable[J];
      if (Entry.Base.name)
        delete[] Entry.Base.name;
    }

    OffloadTables[DeviceId].clear();
    return false;
  }

  if (DebugLevel > 0) {
    IDP("Device offload table loaded:\n");
    for (size_t I = 0; I < DeviceNumEntries; ++I)
      IDP("\t%zu:\t%s\n", I, DeviceTable[I].Base.name);
  }

  return true;
}

void RTLDeviceInfoTy::unloadOffloadTable(int32_t DeviceId) {
  for (auto &E : OffloadTables[DeviceId])
    delete[] E.Base.name;

  OffloadTables[DeviceId].clear();
}

void *RTLDeviceInfoTy::allocDataClMem(int32_t DeviceId, size_t Size) {
  cl_mem ret = nullptr;
  cl_int rc;

  CALL_CL_RVRC(ret, clCreateBuffer, rc, getContext(DeviceId),
               CL_MEM_READ_WRITE, Size, nullptr);
  if (rc != CL_SUCCESS)
    return nullptr;

  std::unique_lock<std::mutex> lock(Mutexes[DeviceId]);
  ClMemBuffers[DeviceId].insert((void *)ret);

  IDP("Allocated cl_mem data " DPxMOD "\n", DPxPTR(ret));
  return (void *)ret;
}


void SpecConstantsTy::setProgramConstants(
    int32_t DeviceId, cl_program Program) const {
  cl_int Rc;

  if (!DeviceInfo->isExtensionFunctionEnabled(
          DeviceId, clSetProgramSpecializationConstantId)) {
    IDP("Error: Extension %s is not supported.\n",
        DeviceInfo->getExtensionFunctionName(
            DeviceId, clSetProgramSpecializationConstantId));
    return;
  }

  for (int I = ConstantValues.size(); I > 0; --I) {
    cl_uint Id = static_cast<cl_uint>(ConstantIds[I - 1]);
    size_t Size = ConstantValueSizes[I - 1];
    const void *Val = ConstantValues[I - 1];
    CALL_CL_EXT_SILENT(DeviceId, Rc, clSetProgramSpecializationConstant,
                       Program, Id, Size, Val);
    if (Rc == CL_SUCCESS)
      IDP("Set specialization constant '0x%X'\n", static_cast<int32_t>(Id));
  }
}

#ifdef __cplusplus
extern "C" {
#endif

static inline void addDataTransferLatency() {
  if (!DeviceInfo->Flags.CollectDataTransferLatency)
    return;
  double goal = omp_get_wtime() + 1e-6 * DeviceInfo->DataTransferLatency;
  // Naive spinning should be enough
  while (omp_get_wtime() < goal)
    ;
}

int32_t ExtensionsTy::getExtensionsInfoForDevice(int32_t DeviceNum) {
  // Identify the size of OpenCL extensions string.
  size_t RetSize = 0;
  // If the below call fails, some extensions's status may be
  // left ExtensionStatusUnknown, so only ExtensionStatusEnabled
  // actually means that the extension is enabled.
  IDP("Getting extensions for device %d\n", DeviceNum);

  cl_device_id DeviceId = DeviceInfo->deviceIDs[DeviceNum];
  CALL_CL_RET_FAIL(clGetDeviceInfo, DeviceId, CL_DEVICE_EXTENSIONS, 0, nullptr,
                   &RetSize);

  std::unique_ptr<char []> Data(new char[RetSize]);
  CALL_CL_RET_FAIL(clGetDeviceInfo, DeviceId, CL_DEVICE_EXTENSIONS, RetSize,
                   Data.get(), &RetSize);

  std::string Extensions(Data.get());
  IDP("Device extensions: %s\n", Extensions.c_str());

  if (UnifiedSharedMemory == ExtensionStatusUnknown &&
      Extensions.find("cl_intel_unified_shared_memory") != std::string::npos) {
    UnifiedSharedMemory = ExtensionStatusEnabled;
    IDP("Extension UnifiedSharedMemory enabled.\n");
  }

  std::for_each(LibdeviceExtensions.begin(), LibdeviceExtensions.end(),
                [&Extensions](LibdeviceExtDescTy &Desc) {
                  if (Desc.Status == ExtensionStatusUnknown)
                    if (Extensions.find(Desc.Name) != std::string::npos) {
                      Desc.Status = ExtensionStatusEnabled;
                      IDP("Extension %s enabled.\n", Desc.Name);
                    }
                });

  return CL_SUCCESS;
}

// FIXME: move this to llvm/BinaryFormat/ELF.h and elf.h:
#define NT_INTEL_ONEOMP_OFFLOAD_VERSION 1
#define NT_INTEL_ONEOMP_OFFLOAD_IMAGE_COUNT 2
#define NT_INTEL_ONEOMP_OFFLOAD_IMAGE_AUX 3

static bool isValidOneOmpImage(__tgt_device_image *Image,
                               uint64_t &MajorVer,
                               uint64_t &MinorVer) {
  char *ImgBegin = reinterpret_cast<char *>(Image->ImageStart);
  char *ImgEnd = reinterpret_cast<char *>(Image->ImageEnd);
  size_t ImgSize = ImgEnd - ImgBegin;
  ElfL E(ImgBegin, ImgSize);
  if (!E.isValidElf()) {
    DP("Warning: unable to get ELF handle: %s!\n", E.getErrmsg(-1));
    return false;
  }

  for (auto I = E.section_notes_begin(), IE = E.section_notes_end(); I != IE;
       ++I) {
    ElfLNote Note = *I;
    if (Note.getNameSize() == 0)
      continue;
    std::string NameStr(Note.getName(), Note.getNameSize());
    if (NameStr != "INTELONEOMPOFFLOAD")
      continue;
    uint64_t Type = Note.getType();
    if (Type != NT_INTEL_ONEOMP_OFFLOAD_VERSION)
      continue;
    std::string DescStr(reinterpret_cast<const char *>(Note.getDesc()),
                        Note.getDescSize());
    auto DelimPos = DescStr.find('.');
    if (DelimPos == std::string::npos) {
      // The version has to look like "Major#.Minor#".
      DP("Invalid NT_INTEL_ONEOMP_OFFLOAD_VERSION: '%s'\n", DescStr.c_str());
      return false;
    }
    std::string MajorVerStr = DescStr.substr(0, DelimPos);
    DescStr.erase(0, DelimPos + 1);
    MajorVer = std::stoull(MajorVerStr);
    MinorVer = std::stoull(DescStr);
    bool isSupported = (MajorVer == 1 && MinorVer == 0);
    return isSupported;
  }

  return false;
}

EXTERN
int32_t __tgt_rtl_is_valid_binary(__tgt_device_image *Image) {
  uint64_t MajorVer, MinorVer;
  if (isValidOneOmpImage(Image, MajorVer, MinorVer)) {
    DP("Target binary is a valid oneAPI OpenMP image.\n");
    return 1;
  }

  DP("Target binary is *not* a valid oneAPI OpenMP image.\n");

  // Fallback to legacy behavior, when the image is a plain
  // SPIR-V file.
  uint32_t MagicWord = *(uint32_t *)Image->ImageStart;
  // compare magic word in little endian and big endian:
  int32_t Ret = (MagicWord == 0x07230203 || MagicWord == 0x03022307);
  IDP("Target binary is %s\n", Ret ? "VALID" : "INVALID");

  return Ret;
}

/// Convert device name to device arch.
static uint64_t getDeviceArch(const char *DeviceName) {
  if (DeviceInfo->DeviceType == CL_DEVICE_TYPE_CPU)
    return DeviceArch_x86_64;

  std::string name(DeviceName);
#ifdef _WIN32
  // Windows: Device name contains published product name.
  for (auto &arch : DeviceArchMap)
    for (auto str : arch.second)
      if (name.find(str) != std::string::npos)
        return arch.first;
#else
  // Linux: Device name contains "[0xABCD]" device identifier.
  auto pos = name.rfind("[");
  uint32_t OCLDeviceId = 0;

  if (pos != std::string::npos && name.size() - pos >= 8)
    OCLDeviceId = std::strtol(name.substr(pos + 1, 6).c_str(), nullptr, 16);

  if (OCLDeviceId != 0) {
    for (auto &arch : DeviceArchMap)
      for (auto id : arch.second)
        if (OCLDeviceId == id || (OCLDeviceId & 0xFF00) == id)
          return arch.first;  // Exact match or prefix match
  }
#endif

  IDP("Warning: Cannot decide device arch for %s.\n", DeviceName);
  return DeviceArch_None;
}

EXTERN
int32_t __tgt_rtl_number_of_devices() {
  // Assume it is thread safe, since it is called once.

  IDP("Start initializing OpenCL\n");
  // get available platforms
  cl_uint platformIdCount = 0;
  CALL_CL_RET_ZERO(clGetPlatformIDs, 0, nullptr, &platformIdCount);
  std::vector<cl_platform_id> platformIds(platformIdCount);
  CALL_CL_RET_ZERO(clGetPlatformIDs, platformIdCount, platformIds.data(),
                   nullptr);

  // All eligible OpenCL device IDs from the platforms are stored in a list
  // in the order they are probed by clGetPlatformIDs/clGetDeviceIDs.
  for (cl_platform_id id : platformIds) {
    std::vector<char> buf;
    size_t buf_size;
    cl_int rc;
    CALL_CL(rc, clGetPlatformInfo, id, CL_PLATFORM_VERSION, 0, nullptr,
            &buf_size);
    if (rc != CL_SUCCESS || buf_size == 0)
      continue;
    buf.resize(buf_size);
    CALL_CL(rc, clGetPlatformInfo, id, CL_PLATFORM_VERSION, buf_size,
            buf.data(), nullptr);
    // clCreateProgramWithIL() requires OpenCL 2.1.
    if (rc != CL_SUCCESS || std::stof(std::string(buf.data() + 6)) <= 2.0) {
      continue;
    }
    cl_uint numDevices = 0;
    CALL_CL_SILENT(rc, clGetDeviceIDs, id, DeviceInfo->DeviceType, 0, nullptr,
                   &numDevices);
    if (rc != CL_SUCCESS || numDevices == 0)
      continue;

    const char *platformName = buf.data() ? buf.data() : "undefined";
    IDP("Platform %s has %" PRIu32 " Devices\n", platformName, numDevices);
    std::vector<cl_device_id> devices(numDevices);
    CALL_CL_RET_ZERO(clGetDeviceIDs, id, DeviceInfo->DeviceType, numDevices,
                     devices.data(), nullptr);

    cl_context context = nullptr;
    if (DeviceInfo->Flags.UseSingleContext) {
      cl_context_properties contextProperties[] = {
          CL_CONTEXT_PLATFORM, (cl_context_properties)id, 0
      };
      CALL_CL_RVRC(context, clCreateContext, rc, contextProperties,
                   devices.size(), devices.data(), nullptr, nullptr);
      if (rc != CL_SUCCESS)
        continue;
    }

    DeviceInfo->PlatformInfos.emplace(id, PlatformInfoTy(id, context));
    for (auto device : devices) {
      DeviceInfo->deviceIDs.push_back(device);
      DeviceInfo->Platforms.push_back(id);
    }
    DeviceInfo->numDevices += numDevices;
  }

  if (!DeviceInfo->Flags.UseSingleContext)
    DeviceInfo->Contexts.resize(DeviceInfo->numDevices);
  DeviceInfo->maxExecutionUnits.resize(DeviceInfo->numDevices);
  DeviceInfo->maxWorkGroupSize.resize(DeviceInfo->numDevices);
  DeviceInfo->Extensions.resize(DeviceInfo->numDevices);
  DeviceInfo->Queues.resize(DeviceInfo->numDevices);
  DeviceInfo->QueuesInOrder.resize(DeviceInfo->numDevices, nullptr);
  DeviceInfo->FuncGblEntries.resize(DeviceInfo->numDevices);
  DeviceInfo->KernelProperties.resize(DeviceInfo->numDevices);
  DeviceInfo->Buffers.resize(DeviceInfo->numDevices);
  DeviceInfo->ClMemBuffers.resize(DeviceInfo->numDevices);
  DeviceInfo->ImplicitArgs.resize(DeviceInfo->numDevices);
  DeviceInfo->Profiles.resize(DeviceInfo->numDevices);
  DeviceInfo->Names.resize(DeviceInfo->numDevices);
  DeviceInfo->DeviceArchs.resize(DeviceInfo->numDevices);
  DeviceInfo->Initialized.resize(DeviceInfo->numDevices);
  DeviceInfo->SLMSize.resize(DeviceInfo->numDevices);
  if (DeviceInfo->Flags.UseSVM && DeviceInfo->DeviceType == CL_DEVICE_TYPE_CPU)
    DeviceInfo->DeviceAccessibleData.resize(DeviceInfo->numDevices);
  DeviceInfo->Mutexes = new std::mutex[DeviceInfo->numDevices];
  DeviceInfo->ProfileLocks = new std::mutex[DeviceInfo->numDevices];
  DeviceInfo->OffloadTables.resize(DeviceInfo->numDevices);
  DeviceInfo->OwnedMemory.resize(DeviceInfo->numDevices);

  // get device specific information
  for (unsigned i = 0; i < DeviceInfo->numDevices; i++) {
    size_t buf_size;
    cl_int rc;
    cl_device_id deviceId = DeviceInfo->deviceIDs[i];
    CALL_CL(rc, clGetDeviceInfo, deviceId, CL_DEVICE_NAME, 0, nullptr,
            &buf_size);
    if (rc != CL_SUCCESS || buf_size == 0)
      continue;
    DeviceInfo->Names[i].resize(buf_size);
    CALL_CL(rc, clGetDeviceInfo, deviceId, CL_DEVICE_NAME, buf_size,
            DeviceInfo->Names[i].data(), nullptr);
    if (rc != CL_SUCCESS)
      continue;
    DeviceInfo->DeviceArchs[i] = getDeviceArch(DeviceInfo->Names[i].data());
    IDP("Device %d: %s\n", i, DeviceInfo->Names[i].data());
    CALL_CL_RET_ZERO(clGetDeviceInfo, deviceId, CL_DEVICE_MAX_COMPUTE_UNITS, 4,
                     &DeviceInfo->maxExecutionUnits[i], nullptr);
    IDP("Number of execution units on the device is %d\n",
       DeviceInfo->maxExecutionUnits[i]);
    CALL_CL_RET_ZERO(clGetDeviceInfo, deviceId, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                     sizeof(size_t), &DeviceInfo->maxWorkGroupSize[i], nullptr);
    IDP("Maximum work group size for the device is %d\n",
       static_cast<int32_t>(DeviceInfo->maxWorkGroupSize[i]));
    cl_uint addressmode;
    CALL_CL_RET_ZERO(clGetDeviceInfo, deviceId, CL_DEVICE_ADDRESS_BITS, 4,
                     &addressmode, nullptr);
    IDP("Addressing mode is %d bit\n", addressmode);
    CALL_CL_RET_ZERO(clGetDeviceInfo, deviceId, CL_DEVICE_LOCAL_MEM_SIZE,
                     sizeof(cl_ulong), &DeviceInfo->SLMSize[i], nullptr);
    IDP("Device local mem size: %zu\n", (size_t)DeviceInfo->SLMSize[i]);
    DeviceInfo->Initialized[i] = false;
  }
  if (DeviceInfo->numDevices == 0) {
    IDP("WARNING: No OpenCL devices found.\n");
  }

#ifndef _WIN32
  // Make sure it is registered after OCL handlers are registered.
  // Registerization is done in DLLmain for Windows
  if (std::atexit(closeRTL)) {
    FATAL_ERROR("Registration of clean-up function");
  }
#endif //WIN32

  return DeviceInfo->numDevices;
}

EXTERN
int32_t __tgt_rtl_init_device(int32_t device_id) {
  cl_int status;
  IDP("Initialize OpenCL device\n");
  assert(device_id >= 0 && (cl_uint)device_id < DeviceInfo->numDevices &&
         "bad device id");

  // Use out-of-order queue by default.
  std::vector<cl_queue_properties> qProperties {
      CL_QUEUE_PROPERTIES,
      CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
  };
  if (DeviceInfo->Flags.EnableProfile)
    qProperties.back() |= CL_QUEUE_PROFILING_ENABLE;
  qProperties.push_back(0);

  if (!DeviceInfo->Flags.UseSingleContext) {
    auto platform = DeviceInfo->Platforms[device_id];
    auto device = DeviceInfo->deviceIDs[device_id];
    cl_context_properties contextProperties[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0
    };
    cl_int rc;
    CALL_CL_RVRC(DeviceInfo->Contexts[device_id], clCreateContext, rc,
                 contextProperties, 1, &device, nullptr, nullptr);
    if (rc != CL_SUCCESS)
      return OFFLOAD_FAIL;
  }

  auto deviceID = DeviceInfo->deviceIDs[device_id];
  auto context = DeviceInfo->getContext(device_id);
  CALL_CL_RVRC(DeviceInfo->Queues[device_id],
               clCreateCommandQueueWithProperties, status, context, deviceID,
               qProperties.data());
  if (status != CL_SUCCESS) {
    IDP("Error: Failed to create CommandQueue: %d\n", status);
    return OFFLOAD_FAIL;
  }

  DeviceInfo->Extensions[device_id].getExtensionsInfoForDevice(device_id);

  OMPT_CALLBACK(ompt_callback_device_initialize, device_id,
                DeviceInfo->Names[device_id].data(),
                DeviceInfo->deviceIDs[device_id],
                omptLookupEntries, OmptDocument);

  DeviceInfo->Initialized[device_id] = true;

  return OFFLOAD_SUCCESS;
}

EXTERN int64_t __tgt_rtl_init_requires(int64_t RequiresFlags) {
  IDP("Initialize requires flags to %" PRId64 "\n", RequiresFlags);
  DeviceInfo->RequiresFlags = RequiresFlags;
  return RequiresFlags;
}

static void dumpImageToFile(
    const void *Image, size_t ImageSize, const char *Type) { }

static void debugPrintBuildLog(cl_program program, cl_device_id did) { }

static cl_program createProgramFromFile(
    const char *basename, int32_t device_id, std::string &options) {
  std::string device_rtl_path = getDeviceRTLPath(basename);
  std::ifstream device_rtl(device_rtl_path, std::ios::binary);

  if (device_rtl.is_open()) {
    IDP("Found device RTL: %s\n", device_rtl_path.c_str());
    device_rtl.seekg(0, device_rtl.end);
    int device_rtl_len = device_rtl.tellg();
    std::string device_rtl_bin(device_rtl_len, '\0');
    device_rtl.seekg(0);
    if (!device_rtl.read(&device_rtl_bin[0], device_rtl_len)) {
      IDP("I/O Error: Failed to read device RTL.\n");
      return nullptr;
    }

    dumpImageToFile(device_rtl_bin.c_str(), device_rtl_len, basename);

    cl_int status;
    cl_program program;
    CALL_CL_RVRC(program, clCreateProgramWithIL, status,
                 DeviceInfo->getContext(device_id), device_rtl_bin.c_str(),
                 device_rtl_len);
    if (status != CL_SUCCESS) {
      IDP("Error: Failed to create device RTL from IL: %d\n", status);
      return nullptr;
    }

    DeviceInfo->CommonSpecConstants.setProgramConstants(device_id, program);

    CALL_CL(status, clCompileProgram, program, 0, nullptr, options.c_str(), 0,
            nullptr, nullptr, nullptr, nullptr);
    if (status != CL_SUCCESS) {
      debugPrintBuildLog(program, DeviceInfo->deviceIDs[device_id]);
      IDP("Error: Failed to compile program: %d\n", status);
      return nullptr;
    }

    return program;
  }

  IDP("Cannot find device RTL: %s\n", device_rtl_path.c_str());
  return nullptr;
}

static cl_program getOpenCLProgramForImage(int32_t DeviceId,
                                           __tgt_device_image *Image,
                                           std::string &CompilationOptions,
                                           std::string &LinkingOptions,
                                           bool &IsBinary) {
  cl_int Status;

  uint64_t MajorVer, MinorVer;
  if (!isValidOneOmpImage(Image, MajorVer, MinorVer)) {
    // Handle legacy plain SPIR-V image.
    char *ImgBegin = reinterpret_cast<char *>(Image->ImageStart);
    char *ImgEnd = reinterpret_cast<char *>(Image->ImageEnd);
    size_t ImgSize = ImgEnd - ImgBegin;
    dumpImageToFile(ImgBegin, ImgSize, "OpenMP");
    cl_program Program;
    CALL_CL_RVRC(Program, clCreateProgramWithIL, Status,
                 DeviceInfo->getContext(DeviceId),
                 ImgBegin, ImgSize);
    if (Status != CL_SUCCESS) {
      debugPrintBuildLog(Program, DeviceInfo->deviceIDs[DeviceId]);
      IDP("Error: Failed to create program: %d\n", Status);
      return nullptr;
    }

    return Program;
  }

  // Iterate over the images and pick the first one that fits.
  char *ImgBegin = reinterpret_cast<char *>(Image->ImageStart);
  char *ImgEnd = reinterpret_cast<char *>(Image->ImageEnd);
  size_t ImgSize = ImgEnd - ImgBegin;
  ElfL E(ImgBegin, ImgSize);
  assert(E.isValidElf() &&
         "isValidOneOmpImage() returns true for invalid ELF image.");
  assert(MajorVer == 1 && MinorVer == 0 &&
         "FIXME: update image processing for new oneAPI OpenMP version.");
  // Collect auxiliary information.
  uint64_t ImageCount = 0;
  uint64_t MaxImageIdx = 0;
  struct V1ImageInfo {
    // 0 - native, 1 - SPIR-V
    uint64_t Format =  (std::numeric_limits<uint64_t>::max)();
    std::string CompileOpts;
    std::string LinkOpts;
    const uint8_t *Begin;
    uint64_t Size;

    V1ImageInfo(uint64_t Format, std::string CompileOpts,
                std::string LinkOpts, const uint8_t *Begin, uint64_t Size)
      : Format(Format), CompileOpts(CompileOpts),
        LinkOpts(LinkOpts), Begin(Begin), Size(Size) {}
  };

  std::unordered_map<uint64_t, V1ImageInfo> AuxInfo;

  for (auto I = E.section_notes_begin(), IE = E.section_notes_end(); I != IE;
       ++I) {
    ElfLNote Note = *I;
    if (Note.getNameSize() == 0)
      continue;
    std::string NameStr(Note.getName(), Note.getNameSize());
    if (NameStr != "INTELONEOMPOFFLOAD")
      continue;
    uint64_t Type = Note.getType();
    std::string DescStr(reinterpret_cast<const char *>(Note.getDesc()),
                        Note.getDescSize());
    switch (Type) {
    default:
      DP("Warning: unrecognized INTELONEOMPOFFLOAD note.\n");
      break;
    case NT_INTEL_ONEOMP_OFFLOAD_VERSION:
      break;
    case NT_INTEL_ONEOMP_OFFLOAD_IMAGE_COUNT:
      ImageCount = std::stoull(DescStr);
      break;
    case NT_INTEL_ONEOMP_OFFLOAD_IMAGE_AUX: {
      std::vector<std::string> Parts;
      do {
        auto DelimPos = DescStr.find('\0');
        if (DelimPos == std::string::npos) {
          Parts.push_back(DescStr);
          break;
        }
        Parts.push_back(DescStr.substr(0, DelimPos));
        DescStr.erase(0, DelimPos + 1);
      } while (Parts.size() < 4);

      // Ignore records with less than 4 strings.
      if (Parts.size() != 4) {
        DP("Warning: short NT_INTEL_ONEOMP_OFFLOAD_IMAGE_AUX "
           "record is ignored.\n");
        continue;
      }

      uint64_t Idx = std::stoull(Parts[0]);
      MaxImageIdx = (std::max)(MaxImageIdx, Idx);
      if (AuxInfo.find(Idx) != AuxInfo.end()) {
        DP("Warning: duplicate auxiliary information for image %" PRIu64
           " is ignored.\n", Idx);
        continue;
      }
      AuxInfo.emplace(std::piecewise_construct,
                      std::forward_as_tuple(Idx),
                      std::forward_as_tuple(std::stoull(Parts[1]),
                                            Parts[2], Parts[3],
                                            // Image pointer and size
                                            // will be initialized later.
                                            nullptr, 0));
    }
    }
  }

  if (MaxImageIdx >= ImageCount)
    DP("Warning: invalid image index found in auxiliary information.\n");

  for (auto I = E.sections_begin(), IE = E.sections_end(); I != IE; ++I) {
    const char *Prefix = "__openmp_offload_spirv_";
    std::string SectionName((*I).getName() ? (*I).getName() : "");
    if (SectionName.find(Prefix) != 0)
      continue;
    SectionName.erase(0, std::strlen(Prefix));
    uint64_t Idx = std::stoull(SectionName);
    if (Idx >= ImageCount) {
      DP("Warning: ignoring image section (index %" PRIu64
         " is out of range).\n", Idx);
      continue;
    }

    auto AuxInfoIt = AuxInfo.find(Idx);
    if (AuxInfoIt == AuxInfo.end()) {
      DP("Warning: ignoring image section (no aux info).\n");
      continue;
    }

    AuxInfoIt->second.Begin = (*I).getContents();
    AuxInfoIt->second.Size = (*I).getSize();
  }

  for (uint64_t Idx = 0; Idx < ImageCount; ++Idx) {
    auto It = AuxInfo.find(Idx);
    if (It == AuxInfo.end()) {
      DP("Warning: image %" PRIu64
         " without auxiliary information is ingored.\n", Idx);
      continue;
    }

    const unsigned char *ImgBegin =
        reinterpret_cast<const unsigned char *>(It->second.Begin);
    size_t ImgSize = It->second.Size;
    dumpImageToFile(ImgBegin, ImgSize, "OpenMP");
    cl_program Program;

    if (It->second.Format == 0) {
      // Native format.
      IsBinary = true;
      CALL_CL_RVRC(Program, clCreateProgramWithBinary, Status,
                   DeviceInfo->getContext(DeviceId),
                   1, &DeviceInfo->deviceIDs[DeviceId],
                   &ImgSize, &ImgBegin, nullptr);
    } else if (It->second.Format == 1) {
      // SPIR-V format.
      IsBinary = false;
      CALL_CL_RVRC(Program, clCreateProgramWithIL, Status,
                   DeviceInfo->getContext(DeviceId),
                   ImgBegin, ImgSize);
    } else {
      DP("Warning: image %" PRIu64 "is ignored due to unknown format.\n", Idx);
      continue;
    }

    if (Status != CL_SUCCESS) {
      debugPrintBuildLog(Program, DeviceInfo->deviceIDs[DeviceId]);
      DP("Warning: failed to create program from %s (%" PRIu64 "): %d\n",
         IsBinary ? "binary" : "SPIR-V", Idx, Status);
      continue;
    }

    DP("Created offload program from image #%" PRIu64 ".\n", Idx);
    if (DeviceInfo->Flags.UseImageOptions) {
      CompilationOptions += " " + It->second.CompileOpts;
      LinkingOptions += " " + It->second.LinkOpts;
    }
    return Program;
  }

  return nullptr;
}

EXTERN
__tgt_target_table *__tgt_rtl_load_binary(int32_t device_id,
                                          __tgt_device_image *image) {

  IDP("Device %d: load binary from " DPxMOD " image\n", device_id,
     DPxPTR(image->ImageStart));

  size_t NumEntries = (size_t)(image->EntriesEnd - image->EntriesBegin);
  IDP("Expecting to have %zu entries defined.\n", NumEntries);

  ProfileIntervalTy CompilationTimer("Compiling", device_id);
  ProfileIntervalTy LinkingTimer("Linking", device_id);

  // create Program
  cl_int status;
  std::vector<cl_program> programs;
  cl_program linked_program;
  std::string CompilationOptions(
      DeviceInfo->CompilationOptions + " " +
      DeviceInfo->UserCompilationOptions);
  std::string LinkingOptions(DeviceInfo->UserLinkingOptions);
  IDP("Basic OpenCL compilation options: %s\n", CompilationOptions.c_str());
  IDP("Basic OpenCL linking options: %s\n", LinkingOptions.c_str());

  bool IsBinary = false;
  // Create program for the target regions.
  // User program must be first in the link order.
  CompilationTimer.start();
  cl_program program = getOpenCLProgramForImage(device_id, image,
                                                CompilationOptions,
                                                LinkingOptions,
                                                IsBinary);
  if (!program) {
    // This must never happen. The compiler must make sure that
    // at least one generic SPIR-V image fits all devices.
    DP("Error: failed to create a program from the offload image.\n");
    return NULL;
  }

  DeviceInfo->CommonSpecConstants.setProgramConstants(device_id, program);

  IDP("Final OpenCL compilation options: %s\n", CompilationOptions.c_str());
  IDP("Final OpenCL linking options: %s\n", LinkingOptions.c_str());
  // clLinkProgram drops the last symbol. Work this around temporarily.
  LinkingOptions += " ";

  if (IsBinary || DeviceInfo->Flags.EnableSimd ||
      // Work around GPU API issue: clCompileProgram/clLinkProgram
      // does not work with -vc-codegen, so we have to use clBuildProgram.
      CompilationOptions.find(" -vc-codegen ") != std::string::npos) {
    // Programs created from binary must still be built.
    CALL_CL(status, clBuildProgram, program, 0, nullptr,
      (CompilationOptions + " " + LinkingOptions).c_str(), nullptr, nullptr);
    if (status != CL_SUCCESS) {
      debugPrintBuildLog(program, DeviceInfo->deviceIDs[device_id]);
      IDP("Error: Failed to build program: %d\n", status);
      return NULL;
    }
    linked_program = program;
    DeviceInfo->FuncGblEntries[device_id].Program = linked_program;
    CompilationTimer.stop();
  } else {
    CALL_CL(status, clCompileProgram, program, 0, nullptr,
      CompilationOptions.c_str(), 0, nullptr, nullptr, nullptr, nullptr);
    if (status != CL_SUCCESS) {
      debugPrintBuildLog(program, DeviceInfo->deviceIDs[device_id]);
      IDP("Error: Failed to compile program: %d\n", status);
      return NULL;
    }
    programs.push_back(program);

    // Link libdevice fallback implementations, if needed.
    auto &libdevice_extensions =
        DeviceInfo->Extensions[device_id].LibdeviceExtensions;

    for (unsigned i = 0; i < libdevice_extensions.size(); ++i) {
      auto &desc = libdevice_extensions[i];
      if (desc.Status != ExtensionStatusEnabled) {
        // Device runtime does not support this libdevice extension,
        // so we have to link in the fallback implementation.
        //
        // TODO: the device image must specify which libdevice extensions
        //       are actually required. We should link only the required
        //       fallback implementations.
        cl_program program =
            createProgramFromFile(desc.FallbackLibName, device_id,
                                  CompilationOptions);
        if (program)
          programs.push_back(program);
      } else {
        IDP("Skipped device RTL: %s\n", desc.FallbackLibName);
      }
    }
    CompilationTimer.stop();

    LinkingTimer.start();

    CALL_CL_RVRC(linked_program, clLinkProgram, status,
        DeviceInfo->getContext(device_id), 1, &DeviceInfo->deviceIDs[device_id],
        LinkingOptions.c_str(), programs.size(), programs.data(), nullptr,
        nullptr);
    if (status != CL_SUCCESS) {
      debugPrintBuildLog(linked_program, DeviceInfo->deviceIDs[device_id]);
      IDP("Error: Failed to link program: %d\n", status);
      return NULL;
    } else {
      IDP("Successfully linked program.\n");
    }
    DeviceInfo->FuncGblEntries[device_id].Program = linked_program;
    LinkingTimer.stop();
  }

  // create kernel and target entries
  DeviceInfo->FuncGblEntries[device_id].Entries.resize(NumEntries);
  DeviceInfo->FuncGblEntries[device_id].Kernels.resize(NumEntries);
  std::vector<__tgt_offload_entry> &entries =
      DeviceInfo->FuncGblEntries[device_id].Entries;
  std::vector<cl_kernel> &kernels =
      DeviceInfo->FuncGblEntries[device_id].Kernels;

  ProfileIntervalTy EntriesTimer("OffloadEntriesInit", device_id);
  EntriesTimer.start();
  // FIXME: table loading does not work at all on XeLP.
  // Enable it after CMPLRLIBS-33285 is fixed.
  if (DeviceInfo->DeviceArchs[device_id] != DeviceArch_XeLP &&
      !DeviceInfo->loadOffloadTable(device_id, NumEntries))
    IDP("Warning: offload table loading failed.\n");
  EntriesTimer.stop();

  for (unsigned i = 0; i < NumEntries; i++) {
    // Size is 0 means that it is kernel function.
    auto Size = image->EntriesBegin[i].size;

    if (Size != 0) {
      EntriesTimer.start();

      void *HostAddr = image->EntriesBegin[i].addr;
      char *Name = image->EntriesBegin[i].name;

      void *TgtAddr =
          DeviceInfo->getOffloadVarDeviceAddr(device_id, Name, Size);
      if (!TgtAddr) {
        TgtAddr = __tgt_rtl_data_alloc(device_id, Size, HostAddr,
                                       TARGET_ALLOC_DEFAULT);
        __tgt_rtl_data_submit(device_id, TgtAddr, HostAddr, Size);
        IDP("Warning: global variable '%s' allocated. "
          "Direct references will not work properly.\n", Name);
      }

      IDP("Global variable mapped: Name = %s, Size = %zu, "
         "HostPtr = " DPxMOD ", TgtPtr = " DPxMOD "\n",
         Name, Size, DPxPTR(HostAddr), DPxPTR(TgtAddr));
      entries[i].addr = TgtAddr;
      entries[i].name = Name;
      entries[i].size = Size;

      EntriesTimer.stop();
      continue;
    }

    char *name = image->EntriesBegin[i].name;
#if _WIN32
    // FIXME: temporary allow zero padding bytes in the entries table
    //        added by MSVC linker (e.g. for incremental linking).
    if (!name) {
      // Initialize the members to be on the safe side.
      IDP("Warning: Entry with a nullptr name!!!\n");
      entries[i].addr = nullptr;
      entries[i].name = nullptr;
      continue;
    }
#endif  // _WIN32
    CALL_CL_RVRC(kernels[i], clCreateKernel, status, linked_program, name);
    if (status != CL_SUCCESS) {
      // If a kernel was deleted by optimizations (e.g. DCE), then
      // clCreateKernel will fail. We expect that such a kernel
      // will never be actually invoked.
      IDP("Warning: Failed to create kernel %s, %d\n", name, status);
      kernels[i] = nullptr;
    }
    entries[i].addr = &kernels[i];
    entries[i].name = name;

    // Do not try to query information for deleted kernels.
    if (!kernels[i])
      continue;

    // Retrieve kernel group size info.
    size_t kernel_simd_width = 1;
    CALL_CL_RET_NULL(clGetKernelWorkGroupInfo, kernels[i],
                     DeviceInfo->deviceIDs[device_id],
                     CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                     sizeof(size_t), &kernel_simd_width, nullptr);
    DeviceInfo->KernelProperties[device_id][kernels[i]].Width =
        kernel_simd_width;

    size_t kernel_wg_size = 1;
    CALL_CL_RET_NULL(clGetKernelWorkGroupInfo, kernels[i],
                     DeviceInfo->deviceIDs[device_id],
                     CL_KERNEL_WORK_GROUP_SIZE,
                     sizeof(size_t), &kernel_wg_size, nullptr);
    DeviceInfo->KernelProperties[device_id][kernels[i]].MaxThreadGroupSize =
        kernel_wg_size;

    if (DebugLevel > 0) {
      // Show kernel information
      std::vector<char> buf;
      size_t buf_size;
      cl_uint kernel_num_args = 0;
      CALL_CL_RET_NULL(clGetKernelInfo, kernels[i], CL_KERNEL_FUNCTION_NAME, 0,
                       nullptr, &buf_size);
      buf.resize(buf_size);
      CALL_CL_RET_NULL(clGetKernelInfo, kernels[i], CL_KERNEL_FUNCTION_NAME,
                       buf_size, buf.data(), nullptr);
      CALL_CL_RET_NULL(clGetKernelInfo, kernels[i], CL_KERNEL_NUM_ARGS,
                       sizeof(cl_uint), &kernel_num_args, nullptr);
      const char *kernelName = buf.data() ? buf.data() : "undefined";
      IDP("Kernel %d: Name = %s, NumArgs = %d\n", i, kernelName,
          kernel_num_args);
      for (unsigned idx = 0; idx < kernel_num_args; idx++) {
        CALL_CL_RET_NULL(clGetKernelArgInfo, kernels[i], idx,
                         CL_KERNEL_ARG_TYPE_NAME, 0, nullptr, &buf_size);
        buf.resize(buf_size);
        CALL_CL_RET_NULL(clGetKernelArgInfo, kernels[i], idx,
                         CL_KERNEL_ARG_TYPE_NAME, buf_size, buf.data(),
                         nullptr);
        std::string type_name = buf.data();
        CALL_CL_RET_NULL(clGetKernelArgInfo, kernels[i], idx,
                         CL_KERNEL_ARG_NAME, 0, nullptr, &buf_size);
        buf.resize(buf_size);
        CALL_CL_RET_NULL(clGetKernelArgInfo, kernels[i], idx,
                         CL_KERNEL_ARG_NAME, buf_size, buf.data(), nullptr);
        const char *argName = buf.data() ? buf.data() : "undefined";
        IDP("  Arg %2d: %s %s\n", idx, type_name.c_str(), argName);
      }
    }
  }

  // Release intermediate programs and store the final program.
  if (!DeviceInfo->Flags.EnableSimd) {
    for (uint32_t i = 0; i < programs.size(); i++) {
      CALL_CL_EXIT_FAIL(clReleaseProgram, programs[i]);
    }
  }

  if (DeviceInfo->initProgramData(device_id) != OFFLOAD_SUCCESS)
    return nullptr;
  __tgt_target_table &table = DeviceInfo->FuncGblEntries[device_id].Table;
  table.EntriesBegin = &(entries[0]);
  table.EntriesEnd = &(entries.data()[entries.size()]);

  OMPT_CALLBACK(ompt_callback_device_load, device_id,
                "" /* filename */,
                -1 /* offset_in_file */,
                nullptr /* vma_in_file */,
                table.EntriesEnd - table.EntriesBegin /* bytes */,
                table.EntriesBegin /* host_addr */,
                nullptr /* device_addr */,
                0 /* module_id */);

  return &table;
}

void event_callback_completed(cl_event event, cl_int status, void *data) {
  if (status == CL_SUCCESS) {
    if (!data) {
      FATAL_ERROR("Invalid asynchronous offloading event");
    }
    AsyncDataTy *async_data = (AsyncDataTy *)data;
    AsyncEventTy *async_event = async_data->Event;
    if (!async_event || !async_event->handler || !async_event->arg) {
      FATAL_ERROR("Invalid asynchronous offloading event");
    }

    cl_command_type cmd;
    CALL_CL_EXIT_FAIL(clGetEventInfo, event, CL_EVENT_COMMAND_TYPE, sizeof(cmd),
                      &cmd, nullptr);

    // Release the temporary cl_mem object used in the data operation.
    if (async_data->MemToRelease &&
        (cmd == CL_COMMAND_READ_BUFFER || cmd == CL_COMMAND_WRITE_BUFFER)) {
      CALL_CL_EXIT_FAIL(clReleaseMemObject, async_data->MemToRelease);
    }

    if (DeviceInfo->Flags.EnableProfile) {
      const char *event_name;
      switch (cmd) {
      case CL_COMMAND_NDRANGE_KERNEL:
        event_name = "KernelAsync";
        break;
      case CL_COMMAND_SVM_MEMCPY:
        event_name = "DataAsync";
        break;
      case CL_COMMAND_READ_BUFFER:
        event_name = "DataReadAsync";
        break;
      case CL_COMMAND_WRITE_BUFFER:
        event_name = "DataWriteAsync";
        break;
      default:
        event_name = "OthersAsync";
      }
      DeviceInfo->getProfiles(async_data->DeviceId).update(event_name, event);
    }

    // Libomptarget is responsible for defining the handler and argument.
    IDP("Calling asynchronous offloading event handler " DPxMOD
       " with argument " DPxMOD "\n", DPxPTR(async_event->handler),
       DPxPTR(async_event->arg));
    async_event->handler(async_event->arg);
    delete async_data;
  } else {
    FATAL_ERROR("Failed to complete asynchronous offloading");
  }
}

// Notify the kernel about target pointers that are not explicitly
// passed as arguments, but which are pointing to mapped objects
// that may potentially be accessed in the kernel code (e.g. PTR_AND_OBJ
// objects).
EXTERN
int32_t __tgt_rtl_manifest_data_for_region(
    int32_t device_id,
    void *tgt_entry_ptr,
    void **tgt_ptrs,
    size_t num_ptrs) {

  cl_kernel *kernel = static_cast<cl_kernel *>(tgt_entry_ptr);
  IDP("Stashing %" PRIu64 " implicit arguments for kernel " DPxMOD "\n",
     static_cast<uint64_t>(num_ptrs), DPxPTR(kernel));
  DeviceInfo->Mutexes[device_id].lock();
  DeviceInfo->ImplicitArgs[device_id][*kernel] =
      std::set<void *>(tgt_ptrs, tgt_ptrs + num_ptrs);
  DeviceInfo->Mutexes[device_id].unlock();

  return OFFLOAD_SUCCESS;
}

EXTERN void __tgt_rtl_create_offload_queue(int32_t DeviceId, void *Interop) {
  if (Interop == nullptr) {
    IDP("Invalid interop object in %s\n", __func__);
    return;
  }

  __tgt_interop_obj *obj = static_cast<__tgt_interop_obj *>(Interop);
  auto isAsync = obj->is_async;
  cl_int status;
  cl_command_queue queue = nullptr;
  auto device = DeviceInfo->deviceIDs[DeviceId];
  auto context = DeviceInfo->getContext(DeviceId);

  // Queue properties for profiling
  cl_queue_properties qProperties[] = {
    CL_QUEUE_PROPERTIES,
    CL_QUEUE_PROFILING_ENABLE,
    0
  };
  auto enableProfile = DeviceInfo->Flags.EnableProfile;

  // Return a shared in-order queue for synchronous case if requested
  if (!isAsync && DeviceInfo->Flags.UseInteropQueueInorderSharedSync) {
    std::unique_lock<std::mutex> lock(DeviceInfo->Mutexes[DeviceId]);
    queue = DeviceInfo->QueuesInOrder[DeviceId];
    if (!queue) {
      CALL_CL_RVRC(queue, clCreateCommandQueueWithProperties, status, context,
                   device, enableProfile ? qProperties : nullptr);
      if (status != CL_SUCCESS) {
        IDP("Error: Failed to create interop command queue: %d\n", status);
        obj->queue = nullptr;
        return;
      }
      DeviceInfo->QueuesInOrder[DeviceId] = queue;
    }
    IDP("%s returns a shared in-order queue " DPxMOD "\n", __func__,
        DPxPTR(queue));
    obj->queue = queue;
    return;
  }

  // Return a shared out-of-order queue for asynchronous case by default
  if (isAsync && !DeviceInfo->Flags.UseInteropQueueInorderAsync) {
    queue = DeviceInfo->Queues[DeviceId];
    IDP("%s returns a shared out-of-order queue " DPxMOD "\n", __func__,
        DPxPTR(queue));
    obj->queue = queue;
    return;
  }

  // Return a new in-order queue for other cases
  CALL_CL_RVRC(queue, clCreateCommandQueueWithProperties, status, context,
               device, enableProfile ? qProperties : nullptr);
  if (status != CL_SUCCESS) {
    IDP("Error: Failed to create interop command queue\n");
    obj->queue = nullptr;
    return;
  }
  IDP("%s creates and returns a new in-order queue " DPxMOD "\n", __func__,
      DPxPTR(queue));
  obj->queue = queue;
  return;
}

// Release the command queue if it is a new in-order command queue.
EXTERN int32_t __tgt_rtl_release_offload_queue(int32_t device_id, void *queue) {
  cl_command_queue cmdQueue = (cl_command_queue)queue;
  std::unique_lock<std::mutex> lock(DeviceInfo->Mutexes[device_id]);
  if (cmdQueue != DeviceInfo->QueuesInOrder[device_id] &&
      cmdQueue != DeviceInfo->Queues[device_id]) {
    CALL_CL_RET_FAIL(clReleaseCommandQueue, cmdQueue);
    IDP("%s releases an in-order queue " DPxMOD "\n", __func__, DPxPTR(queue));
  }
  return OFFLOAD_SUCCESS;
}

EXTERN void *__tgt_rtl_get_platform_handle(int32_t device_id) {
  auto context = DeviceInfo->getContext(device_id);
  return (void *) context;
}

EXTERN void __tgt_rtl_set_device_handle(int32_t DeviceId, void *Interop) {
  if (Interop == nullptr) {
    IDP("Invalid interop object in %s\n", __func__);
    return;
  }
  __tgt_interop_obj *obj = static_cast<__tgt_interop_obj *>(Interop);
  obj->device_handle = DeviceInfo->deviceIDs[DeviceId];
}

EXTERN void *__tgt_rtl_get_context_handle(int32_t DeviceId) {
  auto context = DeviceInfo->getContext(DeviceId);
  return (void *)context;
}

// Allocate a managed memory object.
EXTERN void *__tgt_rtl_data_alloc_managed(int32_t device_id, int64_t size) {
  int32_t kind = DeviceInfo->Flags.UseHostMemForUSM ? TARGET_ALLOC_HOST
                                                    : TARGET_ALLOC_SHARED;
  return __tgt_rtl_data_alloc_explicit(device_id, size, kind);
}

// Check if the pointer belongs to a device-accessible pointer ranges
EXTERN int32_t __tgt_rtl_is_device_accessible_ptr(
    int32_t DeviceId, void *Ptr) {
  // What we want here is to check if Ptr is a SVM/USM pointer.
  // For GPU device, use the USM API to allow use of external memory allocation.
  // For CPU device, use the existing internal data when SVM is enabled since
  // USM API does not return consistent result.
  int32_t ret = false;
  if (DeviceInfo->Flags.UseSVM &&
      DeviceInfo->DeviceType == CL_DEVICE_TYPE_CPU) {
    std::unique_lock<std::mutex> lock(DeviceInfo->Mutexes[DeviceId]);
    for (auto &range : DeviceInfo->DeviceAccessibleData[DeviceId]) {
      intptr_t base = (intptr_t)range.first;
      if (base <= (intptr_t)Ptr && (intptr_t)Ptr < base + range.second) {
        ret = true;
        break;
      }
    }
  } else {
    cl_unified_shared_memory_type_intel memType = 0;
    CALL_CL_EXT_RET(DeviceId, false, clGetMemAllocInfoINTEL,
                    DeviceInfo->getContext(DeviceId), Ptr,
                    CL_MEM_ALLOC_TYPE_INTEL, sizeof(memType), &memType,
                    nullptr);
    switch (memType) {
    case CL_MEM_TYPE_HOST_INTEL:
    case CL_MEM_TYPE_DEVICE_INTEL:
    case CL_MEM_TYPE_SHARED_INTEL:    // Includes SVM on GPU
      ret = true;
      break;
    case CL_MEM_TYPE_UNKNOWN_INTEL:   // Normal host memory
    default:
      ret = false;
    }
  }
  IDP("Ptr " DPxMOD " is %sa device-accessible pointer.\n", DPxPTR(Ptr),
      ret ? "" : "not ");
  return ret;
}

static inline void *dataAlloc(int32_t DeviceId, int64_t Size, void *hstPtr,
    void *hostBase, int32_t ImplicitArg) {
  intptr_t offset = (intptr_t)hstPtr - (intptr_t)hostBase;
  // If the offset is negative, then for our practical purposes it can be
  // considered 0 because the base address of an array will be contained
  // within or after the allocated memory.
  intptr_t meaningfulOffset = offset >= 0 ? offset : 0;
  // If the offset is negative and the size we map is not large enough to reach
  // the base, then we must allocate extra memory up to the base (+1 to include
  // at least the first byte the base is pointing to).
  int64_t meaningfulSize =
      offset < 0 && abs(offset) >= Size ? abs(offset) + 1 : Size;

  void *base = nullptr;
  auto context = DeviceInfo->getContext(DeviceId);
  size_t allocSize = meaningfulSize + meaningfulOffset;

  ProfileIntervalTy dataAllocTimer("DataAlloc", DeviceId);
  dataAllocTimer.start();

  if (DeviceInfo->Flags.UseSVM) {
    CALL_CL_RV(base, clSVMAlloc, context, CL_MEM_READ_WRITE, allocSize, 0);
  } else {
    if (!DeviceInfo->isExtensionFunctionEnabled(DeviceId,
                                                clDeviceMemAllocINTELId)) {
      IDP("Error: Extension %s is not supported\n",
          DeviceInfo->getExtensionFunctionName(DeviceId,
                                               clDeviceMemAllocINTELId));
      return nullptr;
    }
    cl_int rc;
    CALL_CL_EXT_RVRC(DeviceId, base, clDeviceMemAllocINTEL, rc, context,
                     DeviceInfo->deviceIDs[DeviceId], nullptr, allocSize, 0);
    if (rc != CL_SUCCESS)
      return nullptr;
  }
  if (!base) {
    IDP("Error: Failed to allocate base buffer\n");
    return nullptr;
  }
  IDP("Created base buffer " DPxMOD " during data alloc\n", DPxPTR(base));

  void *ret = (void *)((intptr_t)base + meaningfulOffset);

  // Store allocation information
  DeviceInfo->Buffers[DeviceId][ret] = {base, meaningfulSize};

  // Store list of pointers to be passed to kernel implicitly
  if (ImplicitArg) {
    IDP("Stashing an implicit argument " DPxMOD " for next kernel\n",
        DPxPTR(ret));
    DeviceInfo->Mutexes[DeviceId].lock();
    if (DeviceInfo->Flags.UseSVM &&
        DeviceInfo->DeviceType == CL_DEVICE_TYPE_CPU)
      DeviceInfo->DeviceAccessibleData[DeviceId].emplace(
          std::make_pair(ret, meaningfulSize));
    // key "0" for kernel-independent implicit arguments
    DeviceInfo->ImplicitArgs[DeviceId][0].insert(ret);
    DeviceInfo->Mutexes[DeviceId].unlock();
  }

  dataAllocTimer.stop();

  return ret;
}

EXTERN void *__tgt_rtl_data_alloc_explicit(
    int32_t device_id, int64_t size, int32_t kind) {
  auto device = DeviceInfo->deviceIDs[device_id];
  auto context = DeviceInfo->getContext(device_id);
  cl_int rc;
  void *mem = nullptr;
  auto &mutex = DeviceInfo->Mutexes[device_id];
  ProfileIntervalTy dataAllocTimer("DataAlloc", device_id);
  dataAllocTimer.start();

  switch (kind) {
  case TARGET_ALLOC_DEVICE:
    mem = dataAlloc(device_id, size, nullptr, nullptr, 0);
    break;
  case TARGET_ALLOC_HOST:
    if (!DeviceInfo->isExtensionFunctionEnabled(device_id,
                                                clHostMemAllocINTELId)) {
      IDP("Host memory allocator is not available\n");
      return nullptr;
    }
    CALL_CL_EXT_RVRC(device_id, mem, clHostMemAllocINTEL,
                     rc, context, nullptr, size, 0);
    if (mem) {
      if (DeviceInfo->Flags.UseSVM &&
          DeviceInfo->DeviceType == CL_DEVICE_TYPE_CPU) {
        std::unique_lock<std::mutex> dataLock(mutex);
        DeviceInfo->DeviceAccessibleData[device_id].emplace(
            std::make_pair(mem, size));
      }
      IDP("Allocated a host memory object " DPxMOD "\n", DPxPTR(mem));
    }
    break;
  case TARGET_ALLOC_SHARED:
    if (!DeviceInfo->isExtensionFunctionEnabled(
          device_id, clSharedMemAllocINTELId)) {
      IDP("Shared memory allocator is not available\n");
      return nullptr;
    }
    CALL_CL_EXT_RVRC(device_id, mem, clSharedMemAllocINTEL,
                     rc, context, device, nullptr, size, 0);
    if (mem) {
      if (DeviceInfo->Flags.UseSVM &&
          DeviceInfo->DeviceType == CL_DEVICE_TYPE_CPU) {
        std::unique_lock<std::mutex> dataLock(mutex);
        DeviceInfo->DeviceAccessibleData[device_id].emplace(
            std::make_pair(mem, size));
      }
      IDP("Allocated a shared memory object " DPxMOD "\n", DPxPTR(mem));
    }
    break;
  default:
    FATAL_ERROR("Invalid target data allocation kind");
  }

  // Add it to implicit arguments
  mutex.lock();
  DeviceInfo->ImplicitArgs[device_id][0].insert(mem);
  mutex.unlock();

  dataAllocTimer.stop();

  return mem;
}

EXTERN
void *__tgt_rtl_data_alloc(int32_t device_id, int64_t size, void *hst_ptr,
                           int32_t Kind) {
  return dataAlloc(device_id, size, hst_ptr, hst_ptr, 0);
}

// Allocate a base buffer with the given information.
EXTERN
void *__tgt_rtl_data_alloc_base(int32_t device_id, int64_t size, void *hst_ptr,
                                void *hst_base) {
  return dataAlloc(device_id, size, hst_ptr, hst_base, 0);
}

// Allocation was initiated by user (omp_target_alloc)
EXTERN void *__tgt_rtl_data_alloc_user(int32_t device_id, int64_t size,
                                       void *hst_ptr) {
  if (DeviceInfo->Flags.UseBuffer)
    return DeviceInfo->allocDataClMem(device_id, size);

  return dataAlloc(device_id, size, hst_ptr, hst_ptr, 1);
}

EXTERN
int32_t __tgt_rtl_data_submit_nowait(int32_t device_id, void *tgt_ptr,
                                     void *hst_ptr, int64_t size,
                                     void *async_event) {
  if (size == 0)
    // All other plugins seem to be handling 0 size gracefully,
    // so we should do as well.
    return OFFLOAD_SUCCESS;

  cl_command_queue queue = DeviceInfo->Queues[device_id];

  // Add synthetic delay for experiments
  addDataTransferLatency();

  AsyncDataTy *async_data = nullptr;
  if (async_event && ((AsyncEventTy *)async_event)->handler) {
    async_data = new AsyncDataTy((AsyncEventTy *)async_event, device_id);
  }

  if (DeviceInfo->Flags.UseBuffer) {
    std::unique_lock<std::mutex> lock(DeviceInfo->Mutexes[device_id]);
    if (DeviceInfo->ClMemBuffers[device_id].count(tgt_ptr) > 0) {
      cl_event event;
      CALL_CL_RET_FAIL(clEnqueueWriteBuffer, queue, (cl_mem)tgt_ptr, CL_FALSE,
                       0, size, hst_ptr, 0, nullptr, &event);
      if (async_data) {
        CALL_CL_RET_FAIL(clSetEventCallback, event, CL_COMPLETE,
                         &event_callback_completed, async_data);
      } else {
        CALL_CL_RET_FAIL(clWaitForEvents, 1, &event);
        if (DeviceInfo->Flags.EnableProfile)
          DeviceInfo->getProfiles(device_id).update("DataWrite", event);
      }

      return OFFLOAD_SUCCESS;
    }
  }

  if (!DeviceInfo->Flags.UseSVM) {
    if (!DeviceInfo->isExtensionFunctionEnabled(device_id,
                                                clEnqueueMemcpyINTELId)) {
      IDP("Error: Extension %s is not supported\n",
          DeviceInfo->getExtensionFunctionName(device_id,
                                               clEnqueueMemcpyINTELId));
      return OFFLOAD_FAIL;
    }
    cl_event event;
    CALL_CL_EXT_RET_FAIL(device_id, clEnqueueMemcpyINTEL, queue, CL_FALSE,
                         tgt_ptr, hst_ptr, size, 0, nullptr, &event);
    if (async_data) {
      CALL_CL_RET_FAIL(clSetEventCallback, event, CL_COMPLETE,
                       &event_callback_completed, async_data);
    } else {
      CALL_CL_RET_FAIL(clWaitForEvents, 1, &event);
      if (DeviceInfo->Flags.EnableProfile)
        DeviceInfo->getProfiles(device_id).update("DataWrite", event);
    }
    return OFFLOAD_SUCCESS;
  }

  switch (DeviceInfo->DataTransferMethod) {
  case DATA_TRANSFER_METHOD_SVMMAP: {
    cl_event event;
    if (async_data) {
      // Use SVMMemcpy for asynchronous transfer
      CALL_CL_RET_FAIL(clEnqueueSVMMemcpy, queue, CL_FALSE, tgt_ptr, hst_ptr,
                       size, 0, nullptr, &event);
      CALL_CL_RET_FAIL(clSetEventCallback, event, CL_COMPLETE,
                       &event_callback_completed, async_data);
    } else {
      ProfileIntervalTy SubmitTime("DataWrite", device_id);
      SubmitTime.start();

      CALL_CL_RET_FAIL(clEnqueueSVMMap, queue, CL_TRUE, CL_MAP_WRITE, tgt_ptr,
                       size, 0, nullptr, nullptr);
      memcpy(tgt_ptr, hst_ptr, size);
      CALL_CL_RET_FAIL(clEnqueueSVMUnmap, queue, tgt_ptr, 0, nullptr, &event);
      CALL_CL_RET_FAIL(clWaitForEvents, 1, &event);

      SubmitTime.stop();
    }
  } break;
  case DATA_TRANSFER_METHOD_SVMMEMCPY: {
    cl_event event;
    if (async_data) {
      CALL_CL_RET_FAIL(clEnqueueSVMMemcpy, queue, CL_FALSE, tgt_ptr, hst_ptr,
                       size, 0, nullptr, &event);
      CALL_CL_RET_FAIL(clSetEventCallback, event, CL_COMPLETE,
                       &event_callback_completed, async_data);
    } else {
      CALL_CL_RET_FAIL(clEnqueueSVMMemcpy, queue, CL_TRUE, tgt_ptr, hst_ptr,
                       size, 0, nullptr, &event);
      if (DeviceInfo->Flags.EnableProfile)
        DeviceInfo->getProfiles(device_id).update("DataWrite", event);
    }
  } break;
  case DATA_TRANSFER_METHOD_CLMEM:
  default: {
    cl_event event;
    cl_int rc;
    cl_mem mem = nullptr;
    CALL_CL_RVRC(mem, clCreateBuffer, rc, DeviceInfo->getContext(device_id),
                 CL_MEM_USE_HOST_PTR, size, tgt_ptr);
    if (rc != CL_SUCCESS) {
      IDP("Error: Failed to create a buffer from a SVM pointer " DPxMOD "\n",
         DPxPTR(tgt_ptr));
      return OFFLOAD_FAIL;
    }
    if (async_data) {
      async_data->MemToRelease = mem;
      CALL_CL_RET_FAIL(clEnqueueWriteBuffer, queue, mem, CL_FALSE, 0, size,
                       hst_ptr, 0, nullptr, &event);
      CALL_CL_RET_FAIL(clSetEventCallback, event, CL_COMPLETE,
                       &event_callback_completed, async_data);
    } else {
      CALL_CL_RET_FAIL(clEnqueueWriteBuffer, queue, mem, CL_FALSE, 0, size,
                       hst_ptr, 0, nullptr, &event);
      CALL_CL_RET_FAIL(clWaitForEvents, 1, &event);
      CALL_CL_RET_FAIL(clReleaseMemObject, mem);
      if (DeviceInfo->Flags.EnableProfile)
        DeviceInfo->getProfiles(device_id).update("DataWrite", event);
    }
  }
  }
  return OFFLOAD_SUCCESS;
}

EXTERN
int32_t __tgt_rtl_data_submit(int32_t device_id, void *tgt_ptr, void *hst_ptr,
                              int64_t size) {
  return __tgt_rtl_data_submit_nowait(device_id, tgt_ptr, hst_ptr, size,
                                      nullptr);
}

EXTERN
int32_t __tgt_rtl_data_submit_async(int32_t device_id, void *tgt_ptr, void *hst_ptr,
                              int64_t size,
                              __tgt_async_info *AsyncInfoPtr /*not used*/) {
  return __tgt_rtl_data_submit_nowait(device_id, tgt_ptr, hst_ptr, size,
                                      nullptr);
}

EXTERN
int32_t __tgt_rtl_data_retrieve_nowait(int32_t device_id, void *hst_ptr,
                                       void *tgt_ptr, int64_t size,
                                       void *async_event) {
  if (size == 0)
    // All other plugins seem to be handling 0 size gracefully,
    // so we should do as well.
    return OFFLOAD_SUCCESS;

  cl_command_queue queue = DeviceInfo->Queues[device_id];

  // Add synthetic delay for experiments
  addDataTransferLatency();

  AsyncDataTy *async_data = nullptr;
  if (async_event && ((AsyncEventTy *)async_event)->handler) {
    async_data = new AsyncDataTy((AsyncEventTy *)async_event, device_id);
  }

  if (DeviceInfo->Flags.UseBuffer) {
    std::unique_lock<std::mutex> lock(DeviceInfo->Mutexes[device_id]);
    if (DeviceInfo->ClMemBuffers[device_id].count(tgt_ptr) > 0) {
      cl_event event;
      CALL_CL_RET_FAIL(clEnqueueReadBuffer, queue, (cl_mem)tgt_ptr, CL_FALSE,
                       0, size, hst_ptr, 0, nullptr, &event);
      if (async_data) {
        CALL_CL_RET_FAIL(clSetEventCallback, event, CL_COMPLETE,
                         &event_callback_completed, async_data);
      } else {
        CALL_CL_RET_FAIL(clWaitForEvents, 1, &event);
        if (DeviceInfo->Flags.EnableProfile)
          DeviceInfo->getProfiles(device_id).update("DataRead", event);
      }

      return OFFLOAD_SUCCESS;
    }
  }

  if (!DeviceInfo->Flags.UseSVM) {
    if (!DeviceInfo->isExtensionFunctionEnabled(device_id,
                                                clEnqueueMemcpyINTELId)) {
      IDP("Error: Extension %s is not supported\n",
          DeviceInfo->getExtensionFunctionName(device_id,
                                               clEnqueueMemcpyINTELId));
      return OFFLOAD_FAIL;
    }
    cl_event event;
    CALL_CL_EXT_RET_FAIL(device_id, clEnqueueMemcpyINTEL, queue, CL_FALSE,
                         hst_ptr, tgt_ptr, size, 0, nullptr, &event);
    if (async_data) {
      CALL_CL_RET_FAIL(clSetEventCallback, event, CL_COMPLETE,
                       &event_callback_completed, async_data);
    } else {
      CALL_CL_RET_FAIL(clWaitForEvents, 1, &event);
      if (DeviceInfo->Flags.EnableProfile)
        DeviceInfo->getProfiles(device_id).update("DataRead", event);
    }
    return OFFLOAD_SUCCESS;
  }

  switch (DeviceInfo->DataTransferMethod) {
  case DATA_TRANSFER_METHOD_SVMMAP: {
    cl_event event;
    if (async_data) {
      // Use SVMMemcpy for asynchronous transfer
      CALL_CL_RET_FAIL(clEnqueueSVMMemcpy, queue, CL_FALSE, hst_ptr, tgt_ptr,
                       size, 0, nullptr, &event);
      CALL_CL_RET_FAIL(clSetEventCallback, event, CL_COMPLETE,
                       &event_callback_completed, async_data);
    } else {
      ProfileIntervalTy RetrieveTime("DataRead", device_id);
      RetrieveTime.start();

      CALL_CL_RET_FAIL(clEnqueueSVMMap, queue, CL_TRUE, CL_MAP_READ, tgt_ptr,
                       size, 0, nullptr, nullptr);
      memcpy(hst_ptr, tgt_ptr, size);
      CALL_CL_RET_FAIL(clEnqueueSVMUnmap, queue, tgt_ptr, 0, nullptr, &event);
      CALL_CL_RET_FAIL(clWaitForEvents, 1, &event);

      RetrieveTime.stop();
    }
  } break;
  case DATA_TRANSFER_METHOD_SVMMEMCPY: {
    cl_event event;
    if (async_data) {
      CALL_CL_RET_FAIL(clEnqueueSVMMemcpy, queue, CL_FALSE, hst_ptr, tgt_ptr,
                       size, 0, nullptr, &event);
      CALL_CL_RET_FAIL(clSetEventCallback, event, CL_COMPLETE,
                       &event_callback_completed, async_data);
    } else {
      CALL_CL_RET_FAIL(clEnqueueSVMMemcpy, queue, CL_TRUE, hst_ptr, tgt_ptr,
                       size, 0, nullptr, &event);
      if (DeviceInfo->Flags.EnableProfile)
        DeviceInfo->getProfiles(device_id).update("DataRead", event);
    }
  } break;
  case DATA_TRANSFER_METHOD_CLMEM:
  default: {
    cl_int rc;
    cl_event event;
    cl_mem mem = nullptr;
    CALL_CL_RVRC(mem, clCreateBuffer, rc, DeviceInfo->getContext(device_id),
                 CL_MEM_USE_HOST_PTR, size, tgt_ptr);
    if (rc != CL_SUCCESS) {
      IDP("Error: Failed to create a buffer from a SVM pointer " DPxMOD "\n",
         DPxPTR(tgt_ptr));
      return OFFLOAD_FAIL;
    }
    if (async_data) {
      CALL_CL_RET_FAIL(clEnqueueReadBuffer, queue, mem, CL_FALSE, 0, size,
                       hst_ptr, 0, nullptr, &event);
      async_data->MemToRelease = mem;
      CALL_CL_RET_FAIL(clSetEventCallback, event, CL_COMPLETE,
                       &event_callback_completed, async_data);
    } else {
      CALL_CL_RET_FAIL(clEnqueueReadBuffer, queue, mem, CL_FALSE, 0, size,
                       hst_ptr, 0, nullptr, &event);
      CALL_CL_RET_FAIL(clWaitForEvents, 1, &event);
      CALL_CL_RET_FAIL(clReleaseMemObject, mem);
      if (DeviceInfo->Flags.EnableProfile)
        DeviceInfo->getProfiles(device_id).update("DataRead", event);
    }
  }
  }
  return OFFLOAD_SUCCESS;
}

EXTERN
int32_t __tgt_rtl_data_retrieve(int32_t device_id, void *hst_ptr, void *tgt_ptr,
                                int64_t size) {
  return __tgt_rtl_data_retrieve_nowait(device_id, hst_ptr, tgt_ptr, size,
                                        nullptr);
}

EXTERN
int32_t
__tgt_rtl_data_retrieve_async(int32_t device_id, void *hst_ptr, void *tgt_ptr,
                              int64_t size,
                              __tgt_async_info *AsyncInfoPtr /*not used*/) {
  return __tgt_rtl_data_retrieve_nowait(device_id, hst_ptr, tgt_ptr, size,
                                        nullptr);
}

EXTERN int32_t __tgt_rtl_is_data_exchangable(int32_t SrcId, int32_t DstId) {
  // Only support this case. We don't have any documented OpenCL behavior for
  // cross-device data transfer.
  if (SrcId == DstId)
    return 1;

  return 0;
}

EXTERN int32_t __tgt_rtl_data_exchange(
    int32_t SrcId, void *SrcPtr, int32_t DstId, void *DstPtr, int64_t Size) {
  if (SrcId != DstId)
    return OFFLOAD_FAIL;

  // This is OK for same-device copy with SVM or USM extension.
  return __tgt_rtl_data_submit(DstId, DstPtr, SrcPtr, Size);
}

EXTERN int32_t __tgt_rtl_data_delete(int32_t DeviceId, void *TgtPtr) {
  void *base = TgtPtr;

  // Internal allocation may have different base pointer, so look it up first
  auto &buffers = DeviceInfo->Buffers[DeviceId];

  DeviceInfo->Mutexes[DeviceId].lock();

  // Deallocate cl_mem data
  if (DeviceInfo->Flags.UseBuffer) {
    auto &clMemBuffers = DeviceInfo->ClMemBuffers[DeviceId];
    if (clMemBuffers.count(TgtPtr) > 0) {
      clMemBuffers.erase(TgtPtr);
      CALL_CL_RET_FAIL(clReleaseMemObject, (cl_mem)TgtPtr);
      return OFFLOAD_SUCCESS;
    }
  }

  // Retrieve base pointer and erase buffer information
  bool hasBufferInfo = false;
  if (buffers.count(TgtPtr) > 0) {
    base = buffers[TgtPtr].Base;
    buffers.erase(TgtPtr);
    hasBufferInfo = true;
  }

  // Erase from the internal list
  for (auto &J : DeviceInfo->ImplicitArgs[DeviceId])
    J.second.erase(TgtPtr);

  DeviceInfo->Mutexes[DeviceId].unlock();

  auto context = DeviceInfo->getContext(DeviceId);
  if (DeviceInfo->Flags.UseSVM && hasBufferInfo) {
    CALL_CL_VOID(clSVMFree, context, base);
  } else {
    CALL_CL_EXT_VOID(DeviceId, clMemFreeINTEL, context, base);
  }

  return OFFLOAD_SUCCESS;
}

static void decideLoopKernelGroupArguments(
    int32_t DeviceId, int32_t ThreadLimit, TgtNDRangeDescTy *LoopLevels,
    cl_kernel Kernel, size_t *GroupSizes, size_t *GroupCounts) {

  size_t maxGroupSize = DeviceInfo->maxWorkGroupSize[DeviceId];
  size_t kernelWidth = DeviceInfo->KernelProperties[DeviceId][Kernel].Width;
  IDP("Assumed kernel SIMD width is %zu\n", kernelWidth);

  size_t kernelMaxThreadGroupSize =
      DeviceInfo->KernelProperties[DeviceId][Kernel].MaxThreadGroupSize;
  if (kernelMaxThreadGroupSize < maxGroupSize) {
    maxGroupSize = kernelMaxThreadGroupSize;
    IDP("Capping maximum thread group size to %zu due to kernel constraints.\n",
       maxGroupSize);
  }

  bool maxGroupSizeForced = false;

  if (ThreadLimit > 0) {
    maxGroupSizeForced = true;

    if ((uint32_t)ThreadLimit <= maxGroupSize) {
      maxGroupSize = ThreadLimit;
      IDP("Max group size is set to %zu (thread_limit clause)\n",
         maxGroupSize);
    } else {
      IDP("thread_limit(%" PRIu32 ") exceeds current maximum %zu\n",
         ThreadLimit, maxGroupSize);
    }
  }

  if (DeviceInfo->ThreadLimit > 0) {
    maxGroupSizeForced = true;

    if (DeviceInfo->ThreadLimit <= maxGroupSize) {
      maxGroupSize = DeviceInfo->ThreadLimit;
      IDP("Max group size is set to %zu (OMP_THREAD_LIMIT)\n",
         maxGroupSize);
    } else {
      IDP("OMP_THREAD_LIMIT(%" PRIu32 ") exceeds current maximum %zu\n",
         DeviceInfo->ThreadLimit, maxGroupSize);
    }
  }

  if (DeviceInfo->NumTeams > 0)
    IDP("OMP_NUM_TEAMS(%" PRIu32 ") is ignored\n", DeviceInfo->NumTeams);

  GroupCounts[0] = GroupCounts[1] = GroupCounts[2] = 1;
  size_t groupSizes[3] = {maxGroupSize, 1, 1};
  TgtLoopDescTy *level = LoopLevels->Levels;
  int32_t distributeDim = LoopLevels->DistributeDim;
  assert(distributeDim >= 0 && distributeDim <= 2 &&
         "Invalid distribute dimension.");
  int32_t numLoopLevels = LoopLevels->NumLoops;
  assert((numLoopLevels > 0 && numLoopLevels <= 3) &&
         "Invalid loop nest description for ND partitioning");

  // Compute global widths for X/Y/Z dimensions.
  size_t tripCounts[3] = {1, 1, 1};

  for (int32_t i = 0; i < numLoopLevels; i++) {
    assert(level[i].Stride > 0 && "Invalid loop stride for ND partitioning");
    IDP("Level %" PRIu32 ": Lb = %" PRId64 ", Ub = %" PRId64 ", Stride = %"
       PRId64 "\n", i, level[i].Lb, level[i].Ub, level[i].Stride);
    if (level[i].Ub < level[i].Lb) {
      std::fill(GroupCounts, GroupCounts + 3, 1);
      std::fill(GroupSizes, GroupSizes + 3, 1);
      return;
    }
    tripCounts[i] =
        (level[i].Ub - level[i].Lb + level[i].Stride) / level[i].Stride;
  }

  if (!maxGroupSizeForced) {
    // Use clGetKernelSuggestedLocalWorkSizeINTEL to compute group sizes,
    // or fallback to setting dimension 0 width to SIMDWidth.
    // Note that in case of user-specified LWS groupSizes[0]
    // is already set according to the specified value.
    size_t globalSizes[3] = { tripCounts[0], tripCounts[1], tripCounts[2] };
    if (distributeDim > 0) {
      // There is a distribute dimension.
      globalSizes[distributeDim - 1] *= globalSizes[distributeDim];
      globalSizes[distributeDim] = 1;
    }

    cl_int rc = CL_DEVICE_NOT_FOUND;
    size_t suggestedGroupSizes[3] = {1, 1, 1};
    if (rc == CL_SUCCESS) {
      groupSizes[0] = suggestedGroupSizes[0];
      groupSizes[1] = suggestedGroupSizes[1];
      groupSizes[2] = suggestedGroupSizes[2];
    } else if (maxGroupSize > kernelWidth) {
      groupSizes[0] = kernelWidth;
    }
  }

  for (int32_t i = 0; i < numLoopLevels; i++) {
    if (i < distributeDim) {
      GroupCounts[i] = 1;
      continue;
    }
    size_t trip = tripCounts[i];
    if (groupSizes[i] >= trip)
      groupSizes[i] = trip;
    GroupCounts[i] = (trip + groupSizes[i] - 1) / groupSizes[i];
  }
  std::copy(groupSizes, groupSizes + 3, GroupSizes);
}

static void decideKernelGroupArguments(
    int32_t DeviceId, int32_t NumTeams, int32_t ThreadLimit,
    cl_kernel Kernel, size_t *GroupSizes, size_t *GroupCounts) {
  size_t maxGroupSize = DeviceInfo->maxWorkGroupSize[DeviceId];
  bool maxGroupSizeForced = false;
  bool maxGroupCountForced = false;

  size_t kernelWidth = DeviceInfo->KernelProperties[DeviceId][Kernel].Width;
  IDP("Assumed kernel SIMD width is %zu\n", kernelWidth);

  size_t kernelMaxThreadGroupSize =
      DeviceInfo->KernelProperties[DeviceId][Kernel].MaxThreadGroupSize;
  if (kernelMaxThreadGroupSize < maxGroupSize) {
    maxGroupSize = kernelMaxThreadGroupSize;
    IDP("Capping maximum thread group size to %zu due to kernel constraints.\n",
       maxGroupSize);
  }

  if (ThreadLimit > 0) {
    maxGroupSizeForced = true;

    if ((uint32_t)ThreadLimit <= maxGroupSize) {
      maxGroupSize = ThreadLimit;
      IDP("Max group size is set to %zu (thread_limit clause)\n",
         maxGroupSize);
    } else {
      IDP("thread_limit(%" PRIu32 ") exceeds current maximum %zu\n",
         ThreadLimit, maxGroupSize);
    }
  }

  if (DeviceInfo->ThreadLimit > 0) {
    maxGroupSizeForced = true;

    if (DeviceInfo->ThreadLimit <= maxGroupSize) {
      maxGroupSize = DeviceInfo->ThreadLimit;
      IDP("Max group size is set to %zu (OMP_THREAD_LIMIT)\n",
         maxGroupSize);
    } else {
      IDP("OMP_THREAD_LIMIT(%" PRIu32 ") exceeds current maximum %zu\n",
         DeviceInfo->ThreadLimit, maxGroupSize);
    }
  }

  size_t maxGroupCount = 0;

  if (NumTeams > 0) {
    maxGroupCount = NumTeams;
    maxGroupCountForced = true;
    IDP("Max group count is set to %zu "
       "(num_teams clause or no teams construct)\n", maxGroupCount);
  } else if (DeviceInfo->NumTeams > 0) {
    // OMP_NUM_TEAMS only matters, if num_teams() clause is absent.
    maxGroupCount = DeviceInfo->NumTeams;
    maxGroupCountForced = true;
    IDP("Max group count is set to %zu (OMP_NUM_TEAMS)\n",
       maxGroupCount);
  }

  if (maxGroupCountForced) {
    // If number of teams is specified by the user, then use kernelWidth
    // WIs per WG by default, so that it matches
    // decideLoopKernelGroupArguments() behavior.
    if (!maxGroupSizeForced) {
      maxGroupSize = kernelWidth;
    }
  } else {
    maxGroupCount = DeviceInfo->maxExecutionUnits[DeviceId];
  }

  GroupSizes[0] = maxGroupSize;
  GroupSizes[1] = GroupSizes[2] = 1;

  GroupCounts[0] = maxGroupCount;
  GroupCounts[1] = GroupCounts[2] = 1;
  if (!maxGroupCountForced)
    GroupCounts[0] *= DeviceInfo->SubscriptionRate;
}

static inline int32_t run_target_team_nd_region(
    int32_t device_id, void *tgt_entry_ptr, void **tgt_args,
    ptrdiff_t *tgt_offsets, int32_t num_args, int32_t num_teams,
    int32_t thread_limit, void *loop_desc, void *async_event) {

  cl_kernel *kernel = static_cast<cl_kernel *>(tgt_entry_ptr);

  if (!*kernel) {
    REPORT("Failed to invoke deleted kernel.\n");
    return OFFLOAD_FAIL;
  }

  // Decide group sizes and counts
  size_t local_work_size[3] = {1, 1, 1};
  size_t num_work_groups[3] = {1, 1, 1};
  if (loop_desc) {
    decideLoopKernelGroupArguments(device_id, thread_limit,
                                   (TgtNDRangeDescTy *)loop_desc, *kernel,
                                   local_work_size, num_work_groups);
  } else {
    decideKernelGroupArguments(device_id, num_teams, thread_limit,
                               *kernel, local_work_size, num_work_groups);
  }

  size_t global_work_size[3];
  for (int32_t i = 0; i < 3; ++i)
    global_work_size[i] = local_work_size[i] * num_work_groups[i];

  IDP("Group sizes = {%zu, %zu, %zu}\n", local_work_size[0],
     local_work_size[1], local_work_size[2]);
  IDP("Group counts = {%zu, %zu, %zu}\n",
      global_work_size[0] / local_work_size[0],
      global_work_size[1] / local_work_size[1],
      global_work_size[2] / local_work_size[2]);

  // Protect thread-unsafe OpenCL API calls
  DeviceInfo->Mutexes[device_id].lock();

  // Set implicit kernel args
  std::vector<void *> implicit_args;
  // USM Implicit arguments to be reported to the runtime.
  std::vector<void *> implicit_usm_args;

  // Array sections of zero size may result in nullptr target pointer,
  // which will not be accepted by clSetKernelExecInfo, so we should
  // avoid manifesting them.

  // Reserve space in implicit_args to speed up the back_inserter.
  size_t num_implicit_args = 0;
  if (DeviceInfo->ImplicitArgs[device_id].count(*kernel) > 0) {
    num_implicit_args += DeviceInfo->ImplicitArgs[device_id][*kernel].size();
  }

  implicit_args.reserve(num_implicit_args);

  if (DeviceInfo->ImplicitArgs[device_id].count(*kernel) > 0) {
    // kernel-dependent arguments
    std::copy_if(DeviceInfo->ImplicitArgs[device_id][*kernel].begin(),
                 DeviceInfo->ImplicitArgs[device_id][*kernel].end(),
                 std::back_inserter(implicit_args),
                 [] (void *ptr) {
                   return ptr != nullptr;
                 });
  }
  if (DeviceInfo->ImplicitArgs[device_id].count(0) > 0) {
    // kernel-independent arguments
    // Note that these pointers may not be nullptr.
    implicit_args.insert(implicit_args.end(),
        DeviceInfo->ImplicitArgs[device_id][0].begin(),
        DeviceInfo->ImplicitArgs[device_id][0].end());
  }

  // set kernel args
//  std::vector<void *> ptrs(num_args);
  for (int32_t i = 0; i < num_args; ++i) {
    ptrdiff_t offset = tgt_offsets[i];
    const char *ArgType = "Unknown";
    // Offset equal to MAX(ptrdiff_t) means that the argument
    // must be passed as literal, and the offset should be ignored.
    if (offset == (std::numeric_limits<ptrdiff_t>::max)()) {
      intptr_t arg = (intptr_t)tgt_args[i];
      CALL_CL_RET_FAIL(clSetKernelArg, *kernel, i, sizeof(arg), &arg);
      ArgType = "Scalar";
    } else {
      ArgType = "Pointer";
      void *ptr = (void *)((intptr_t)tgt_args[i] + offset);
      if (DeviceInfo->Flags.UseBuffer &&
          DeviceInfo->ClMemBuffers[device_id].count(ptr) > 0) {
        CALL_CL_RET_FAIL(clSetKernelArg, *kernel, i, sizeof(cl_mem), &ptr);
        ArgType = "ClMem";
      } else if (DeviceInfo->Flags.UseSVM) {
        CALL_CL_RET_FAIL(clSetKernelArgSVMPointer, *kernel, i, ptr);
      } else {
        if (!DeviceInfo->isExtensionFunctionEnabled(
                device_id, clSetKernelArgMemPointerINTELId)) {
          IDP("Error: Extension %s is not supported\n",
              DeviceInfo->getExtensionFunctionName(
                  device_id, clSetKernelArgMemPointerINTELId));
          return OFFLOAD_FAIL;
        }
        CALL_CL_EXT_RET_FAIL(
            device_id, clSetKernelArgMemPointerINTEL, *kernel, i, ptr);
      }
    }
    IDP("Kernel %s Arg %d set successfully\n", ArgType, i);
    (void)ArgType;
  }

  bool hasUSMArgDevice = false;
  bool hasUSMArgHost = false;
  bool hasUSMArgShared = false;
  if (DeviceInfo->isExtensionFunctionEnabled(device_id,
                                             clGetMemAllocInfoINTELId)) {
    // Reserve space for USM pointers.
    implicit_usm_args.reserve(num_implicit_args);
    // Move USM pointers into a separate list, since they need to be
    // reported to the runtime using a separate clSetKernelExecInfo call.
    implicit_args.erase(
        std::remove_if(implicit_args.begin(),
                       implicit_args.end(),
                       [&](void *ptr) {
                         cl_unified_shared_memory_type_intel type = 0;
                         CALL_CL_EXT_RET(device_id, false,
                             clGetMemAllocInfoINTEL,
                             DeviceInfo->getContext(device_id), ptr,
                             CL_MEM_ALLOC_TYPE_INTEL,
                             sizeof(cl_unified_shared_memory_type_intel),
                             &type, nullptr);
                         DPI("clGetMemAllocInfoINTEL API returned %d "
                             "for pointer " DPxMOD "\n", type, DPxPTR(ptr));
                         // USM pointers are classified as
                         // CL_MEM_TYPE_DEVICE_INTEL.
                         // SVM pointers (e.g. returned by clSVMAlloc)
                         // are classified as CL_MEM_TYPE_UNKNOWN_INTEL.
                         // We cannot allocate any other pointer type now.
                         if (type == CL_MEM_TYPE_HOST_INTEL)
                           hasUSMArgHost = true;
                         else if (type == CL_MEM_TYPE_DEVICE_INTEL)
                           hasUSMArgDevice = true;
                         else if (type == CL_MEM_TYPE_SHARED_INTEL)
                           hasUSMArgShared = true;
                         else
                           return false;
                         implicit_usm_args.push_back(ptr);
                         return true;
                       }),
        implicit_args.end());
  }

  if (implicit_args.size() > 0) {
    IDP("Calling clSetKernelExecInfo to pass %zu implicit SVM arguments "
       "to kernel " DPxMOD "\n", implicit_args.size(), DPxPTR(kernel));
    CALL_CL_RET_FAIL(clSetKernelExecInfo, *kernel, CL_KERNEL_EXEC_INFO_SVM_PTRS,
                     sizeof(void *) * implicit_args.size(),
                     implicit_args.data());
  }

  if (implicit_usm_args.size() > 0) {
    // Report non-argument USM pointers to the runtime.
    IDP("Calling clSetKernelExecInfo to pass %zu implicit USM arguments "
       "to kernel " DPxMOD "\n", implicit_usm_args.size(), DPxPTR(kernel));
    CALL_CL_RET_FAIL(clSetKernelExecInfo, *kernel,
                     CL_KERNEL_EXEC_INFO_USM_PTRS_INTEL,
                     sizeof(void *) * implicit_usm_args.size(),
                     implicit_usm_args.data());
    // Mark the kernel as supporting indirect USM accesses, otherwise,
    // clEnqueueNDRangeKernel call below will fail.
    cl_bool KernelSupportsUSM = CL_TRUE;
    if (hasUSMArgHost)
      CALL_CL_RET_FAIL(clSetKernelExecInfo, *kernel,
                       CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL,
                       sizeof(cl_bool), &KernelSupportsUSM);
    if (hasUSMArgDevice)
      CALL_CL_RET_FAIL(clSetKernelExecInfo, *kernel,
                       CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL,
                       sizeof(cl_bool), &KernelSupportsUSM);
    if (hasUSMArgShared)
      CALL_CL_RET_FAIL(clSetKernelExecInfo, *kernel,
                       CL_KERNEL_EXEC_INFO_INDIRECT_SHARED_ACCESS_INTEL,
                       sizeof(cl_bool), &KernelSupportsUSM);
  }

  if (OMPT_ENABLED) {
    // Push current work size
    size_t finalNumTeams =
        global_work_size[0] * global_work_size[1] * global_work_size[2];
    size_t finalThreadLimit =
        local_work_size[0] * local_work_size[1] * local_work_size[2];
    finalNumTeams /= finalThreadLimit;
    OmptGlobal->getTrace().pushWorkSize(finalNumTeams, finalThreadLimit);
  }

  cl_event event;
  CALL_CL_RET_FAIL(clEnqueueNDRangeKernel, DeviceInfo->Queues[device_id],
                   *kernel, 3, nullptr, global_work_size,
                   local_work_size, 0, nullptr, &event);

  DeviceInfo->Mutexes[device_id].unlock();

  IDP("Started executing kernel.\n");

  if (async_event) {
    if (((AsyncEventTy *)async_event)->handler) {
      // Add event handler if necessary.
      CALL_CL_RET_FAIL(clSetEventCallback, event, CL_COMPLETE,
          &event_callback_completed,
          new AsyncDataTy((AsyncEventTy *)async_event, device_id));
    } else {
      // Make sure all queued commands finish before the next one starts.
      CALL_CL_RET_FAIL(clEnqueueBarrierWithWaitList,
                       DeviceInfo->Queues[device_id], 0, nullptr, nullptr);
    }
  } else {
    CALL_CL_RET_FAIL(clWaitForEvents, 1, &event);
    if (DeviceInfo->Flags.EnableProfile) {
      std::vector<char> buf;
      size_t buf_size;
      CALL_CL_RET_FAIL(clGetKernelInfo, *kernel, CL_KERNEL_FUNCTION_NAME, 0,
                       nullptr, &buf_size);
      std::string kernel_name("Kernel ");
      if (buf_size > 0) {
        buf.resize(buf_size);
        CALL_CL_RET_FAIL(clGetKernelInfo, *kernel, CL_KERNEL_FUNCTION_NAME,
                         buf.size(), buf.data(), nullptr);
        kernel_name += buf.data();
      }
      DeviceInfo->getProfiles(device_id).update(kernel_name.c_str(), event);
    }
    IDP("Successfully finished kernel execution.\n");
  }

  return OFFLOAD_SUCCESS;
}

EXTERN int32_t
__tgt_rtl_run_target_team_nd_region(int32_t device_id, void *tgt_entry_ptr,
                                    void **tgt_args, ptrdiff_t *tgt_offsets,
                                    int32_t num_args, int32_t num_teams,
                                    int32_t thread_limit, void *loop_desc) {
  return run_target_team_nd_region(device_id, tgt_entry_ptr, tgt_args,
                                   tgt_offsets, num_args, num_teams,
                                   thread_limit, loop_desc, nullptr);
}

EXTERN
int32_t __tgt_rtl_run_target_team_nd_region_nowait(
    int32_t device_id, void *tgt_entry_ptr, void **tgt_args,
    ptrdiff_t *tgt_offsets, int32_t num_args, int32_t num_teams,
    int32_t thread_limit, void *loop_desc, void *async_event) {
  return run_target_team_nd_region(device_id, tgt_entry_ptr, tgt_args,
                                   tgt_offsets, num_args, num_teams,
                                   thread_limit, loop_desc, async_event);
}

EXTERN
int32_t __tgt_rtl_run_target_team_region_nowait(
    int32_t device_id, void *tgt_entry_ptr, void **tgt_args,
    ptrdiff_t *tgt_offsets, int32_t arg_num, int32_t team_num,
    int32_t thread_limit, uint64_t loop_tripcount, void *async_event) {
  return run_target_team_nd_region(device_id, tgt_entry_ptr, tgt_args,
                                   tgt_offsets, arg_num, team_num, thread_limit,
                                   nullptr, async_event);
}

EXTERN
int32_t __tgt_rtl_run_target_region_nowait(int32_t device_id,
                                           void *tgt_entry_ptr, void **tgt_args,
                                           ptrdiff_t *tgt_offsets,
                                           int32_t arg_num, void *async_event) {
  return run_target_team_nd_region(device_id, tgt_entry_ptr, tgt_args,
                                   tgt_offsets, arg_num, 1, 0, nullptr,
                                   async_event);
}

EXTERN
int32_t __tgt_rtl_run_target_team_region(int32_t device_id, void *tgt_entry_ptr,
                                         void **tgt_args,
                                         ptrdiff_t *tgt_offsets,
                                         int32_t arg_num, int32_t team_num,
                                         int32_t thread_limit,
                                         uint64_t loop_tripcount /*not used*/) {
  return run_target_team_nd_region(device_id, tgt_entry_ptr, tgt_args,
                                   tgt_offsets, arg_num, team_num, thread_limit,
                                   nullptr, nullptr);
}

EXTERN
int32_t __tgt_rtl_run_target_team_region_async(
    int32_t device_id, void *tgt_entry_ptr, void **tgt_args,
    ptrdiff_t *tgt_offsets, int32_t arg_num, int32_t team_num,
    int32_t thread_limit, uint64_t loop_tripcount /*not used*/,
    __tgt_async_info *AsyncInfoPtr /*not used*/) {
  return run_target_team_nd_region(device_id, tgt_entry_ptr, tgt_args,
                                   tgt_offsets, arg_num, team_num, thread_limit,
                                   nullptr, nullptr);
}

EXTERN
int32_t __tgt_rtl_run_target_region(int32_t device_id, void *tgt_entry_ptr,
                                    void **tgt_args, ptrdiff_t *tgt_offsets,
                                    int32_t arg_num) {
  // use one team!
  return __tgt_rtl_run_target_team_region(device_id, tgt_entry_ptr, tgt_args,
                                          tgt_offsets, arg_num, 1, 0, 0);
}

EXTERN
int32_t
__tgt_rtl_run_target_region_async(int32_t device_id, void *tgt_entry_ptr,
                                  void **tgt_args, ptrdiff_t *tgt_offsets,
                                  int32_t arg_num,
                                  __tgt_async_info *AsyncInfoPtr /*not used*/) {
  // use one team!
  return __tgt_rtl_run_target_team_region(device_id, tgt_entry_ptr, tgt_args,
                                          tgt_offsets, arg_num, 1, 0, 0);
}

EXTERN char *__tgt_rtl_get_device_name(
    int32_t device_id, char *buffer, size_t buffer_max_size) {
  assert(buffer && "buffer cannot be nullptr.");
  assert(buffer_max_size > 0 && "buffer_max_size cannot be zero.");
  CALL_CL_RET_NULL(clGetDeviceInfo, DeviceInfo->deviceIDs[device_id],
                   CL_DEVICE_NAME, buffer_max_size, buffer, nullptr);
  return buffer;
}

EXTERN int32_t __tgt_rtl_synchronize(int32_t device_id,
                                     __tgt_async_info *async_info_ptr) {
  return OFFLOAD_SUCCESS;
}

EXTERN int32_t __tgt_rtl_get_data_alloc_info(
    int32_t DeviceId, int32_t NumPtrs, void *TgtPtrs, void *AllocInfo) {
  auto &buffers = DeviceInfo->Buffers[DeviceId];
  void **tgtPtrs = static_cast<void **>(TgtPtrs);
  __tgt_memory_info *allocInfo = static_cast<__tgt_memory_info *>(AllocInfo);
  for (int32_t i = 0; i < NumPtrs; i++) {
    if (buffers.count(tgtPtrs[i]) == 0) {
      IDP("%s cannot find allocation information for " DPxMOD "\n", __func__,
          DPxPTR(tgtPtrs[i]));
      return OFFLOAD_FAIL;
    }
    allocInfo[i].Base = buffers[tgtPtrs[i]].Base;
    allocInfo[i].Offset = (uintptr_t)tgtPtrs[i] - (uintptr_t)allocInfo[i].Base;
    allocInfo[i].Size = buffers[tgtPtrs[i]].Size;
  }
  return OFFLOAD_SUCCESS;
}

EXTERN void __tgt_rtl_add_build_options(
    const char *CompileOptions, const char *LinkOptions) {
  if (CompileOptions) {
    auto &compileOptions = DeviceInfo->UserCompilationOptions;
    if (compileOptions.empty()) {
      compileOptions = std::string(CompileOptions) + " ";
    } else {
      IDP("Respecting LIBOMPTARGET_OPENCL_COMPILATION_OPTIONS=%s\n",
          compileOptions.c_str());
    }
  }
  if (LinkOptions) {
    auto &linkOptions = DeviceInfo->UserLinkingOptions;
    if (linkOptions.empty()) {
      linkOptions = std::string(LinkOptions) + " ";
    } else {
      IDP("Respecting LIBOMPTARGET_OPENCL_LINKING_OPTIONS=%s\n",
          linkOptions.c_str());
    }
  }
}

EXTERN int32_t __tgt_rtl_is_supported_device(int32_t DeviceId,
                                             void *DeviceType) {
  if (!DeviceType)
    return true;

  uint64_t deviceArch = DeviceInfo->DeviceArchs[DeviceId];
  int32_t ret = (uint64_t)(deviceArch & (uint64_t)DeviceType) == deviceArch;
  IDP("Device %" PRIu32 " does%s match the requested device types " DPxMOD "\n",
      DeviceId, ret ? "" : " not", DPxPTR(DeviceType));
  return ret;
}

EXTERN void __tgt_rtl_deinit(void) {
  // No-op on Linux
#ifdef _WIN32
  if (DeviceInfo) {
    closeRTL();
    deinit();
  }
#endif // _WIN32
}

EXTERN __tgt_interop *__tgt_rtl_create_interop(
    int32_t DeviceId, int32_t InteropContext, int32_t NumPrefers,
    intptr_t *PreferIDs) {
  // Preference-list is ignored since we cannot have multiple runtimes.
  auto ret = new __tgt_interop();
  ret->FrId = OCLInterop::FrId;
  ret->FrName = OCLInterop::FrName;
  ret->Vendor = OCLInterop::Vendor;
  ret->VendorName = OCLInterop::VendorName;
  ret->DeviceNum = DeviceId;

  auto platform = DeviceInfo->Platforms[DeviceId];
  auto context = DeviceInfo->getContext(DeviceId);
  auto device = DeviceInfo->deviceIDs[DeviceId];

  if (InteropContext == OMP_INTEROP_CONTEXT_TARGET ||
      InteropContext == OMP_INTEROP_CONTEXT_TARGETSYNC) {
    ret->Platform = platform;
    ret->Device = device;
    ret->DeviceContext = context;
  }
  if (InteropContext == OMP_INTEROP_CONTEXT_TARGETSYNC) {
    // Create a new out-of-order queue with profiling enabled.
    cl_queue_properties properties[] = {
      CL_QUEUE_PROPERTIES,
      CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE,
      0
    };
    cl_command_queue cmdQueue = nullptr;
    cl_int rc;
    CALL_CL_RVRC(cmdQueue, clCreateCommandQueueWithProperties, rc, context,
                 device, properties);
    if (rc != CL_SUCCESS) {
      IDP("Error: Failed to create targetsync for interop\n");
      delete ret;
      return nullptr;
    }
    ret->TargetSync = cmdQueue;
  }

  ret->RTLProperty = new OCLInterop::Property();

  return ret;
}

EXTERN int32_t __tgt_rtl_release_interop(
    int32_t DeviceId, __tgt_interop *Interop) {
  if (!Interop || Interop->DeviceNum != (intptr_t)DeviceId ||
      Interop->FrId != OCLInterop::FrId) {
    IDP("Invalid/inconsistent OpenMP interop " DPxMOD "\n", DPxPTR(Interop));
    return OFFLOAD_FAIL;
  }

  if (Interop->TargetSync) {
    auto cmdQueue = static_cast<cl_command_queue>(Interop->TargetSync);
    CALL_CL_RET_FAIL(clFinish, cmdQueue);
    CALL_CL_RET_FAIL(clReleaseCommandQueue, cmdQueue);
  }

  auto OCL = static_cast<OCLInterop::Property *>(Interop->RTLProperty);
  delete OCL;
  delete Interop;

  return OFFLOAD_SUCCESS;
}

EXTERN int32_t __tgt_rtl_use_interop(int32_t DeviceId, __tgt_interop *Interop) {
  if (!Interop || Interop->DeviceNum != (intptr_t)DeviceId ||
      Interop->FrId != OCLInterop::FrId) {
    IDP("Invalid/inconsistent OpenMP interop " DPxMOD "\n", DPxPTR(Interop));
    return OFFLOAD_FAIL;
  }

  if (Interop->TargetSync) {
    auto cmdQueue = static_cast<cl_command_queue>(Interop->TargetSync);
    CALL_CL_RET_FAIL(clFinish, cmdQueue);
  }

  return OFFLOAD_SUCCESS;
}

EXTERN int32_t __tgt_rtl_get_num_interop_properties(int32_t DeviceId) {
  // TODO: decide implementation-defined properties
  return 0;
}

/// Return the value of the requested property
EXTERN int32_t __tgt_rtl_get_interop_property_value(
    int32_t DeviceId, __tgt_interop *Interop, int32_t Ipr, int32_t ValueType,
    size_t Size, void *Value) {

  if (Interop->RTLProperty == nullptr)
    return omp_irc_out_of_range;

  int32_t retCode = omp_irc_success;

  // TODO
  //  switch (Ipr) {
  //default:
  //retCode = omp_irc_out_of_range;
  //}

  return retCode;
}

EXTERN const char *__tgt_rtl_get_interop_property_info(
    int32_t DeviceId, int32_t Ipr, int32_t InfoType) {
  int32_t offset = Ipr - omp_ipr_first;
  if (offset < 0 || (size_t)offset >= OCLInterop::IprNames.size())
    return nullptr;

  if (InfoType == OMP_IPR_INFO_NAME)
    return OCLInterop::IprNames[offset];
  else if (InfoType == OMP_IPR_INFO_TYPE_DESC)
    return OCLInterop::IprTypeDescs[offset];

  return nullptr;
}

EXTERN const char *__tgt_rtl_get_interop_rc_desc(int32_t DeviceId,
                                                 int32_t RetCode) {
  // TODO: decide implementation-defined return code.
  return nullptr;
}

#ifdef __cplusplus
}
#endif

#endif // INTEL_COLLAB
