//===---------- pi_esimd_cpu.cpp - CM Emulation Plugin --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file pi_esimd_cpu.cpp
/// Declarations for CM Emulation Plugin. It is the interface between the
/// device-agnostic SYCL runtime layer and underlying CM Emulation
///
/// \ingroup sycl_pi_esimd_cpu

#include <stdint.h>

#include <CL/sycl/backend_types.hpp>
#include <CL/sycl/detail/accessor_impl.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/export.hpp>
#include <CL/sycl/detail/helpers.hpp>
#include <CL/sycl/detail/host_profiling_info.hpp>
#include <CL/sycl/detail/kernel_desc.hpp>
#include <CL/sycl/detail/type_traits.hpp>
#include <CL/sycl/group.hpp>
#include <CL/sycl/id.hpp>
#include <CL/sycl/kernel.hpp>
#include <CL/sycl/nd_item.hpp>
#include <CL/sycl/range.hpp>

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#ifdef __GNUC__
// Linux
#include <dlfcn.h>
#else
// Windows
#include <windows.h>
#endif

#include "pi_esimd_cpu.hpp"

#define PLACEHOLDER_UNUSED(x) (void)x

// Global variables used in PI_esimd_cpu
// Note we only create a simple pointer variables such that C++ RT won't
// deallocate them automatically at the end of the main program.
// The heap memory allocated for this global variable reclaimed only when
// Sycl RT calls piTearDown().
static sycl::detail::ESIMDEmuPluginOpaqueData *PiESimdDeviceAccess;

// To be compared with ESIMD_EMU_PLUGIN_OPAQUE_DATA_VERSION in device
// interface header file
#define ESIMDEmuPluginDataVersion 0

// To be compared with ESIMD_DEVICE_INTERFACE_VERSION in device
// interface header file
#define ESIMDEmuPluginInterfaceVersion 1

using IDBuilder = sycl::detail::Builder;

// Lambda-call interface definition.
// 'extern "C"' is required as CM supports only C-style function calls
// while kernel is given as lambda function
//
#define LAMBDA_WRAPPER_TMPL(ARGTYPE, TAG, DIMS)                                \
  typedef std::function<void(const ARGTYPE &)> LambdaFunction_##TAG;           \
                                                                               \
  extern "C" struct LambdaWrapper_##TAG {                                      \
    LambdaFunction_##TAG Func;                                                 \
    const sycl::range<DIMS> &LocalSize;                                        \
    const sycl::range<DIMS> &GlobalSize;                                       \
    const sycl::id<DIMS> &GlobalOffset;                                        \
    LambdaWrapper_##TAG(LambdaFunction_##TAG ArgFunc,                          \
                        const sycl::range<DIMS> &ArgLocalSize,                 \
                        const sycl::range<DIMS> &ArgGlobalSize,                \
                        const sycl::id<DIMS> &ArgGlobalOffset)                 \
        : Func(ArgFunc), LocalSize(ArgLocalSize), GlobalSize(ArgGlobalSize),   \
          GlobalOffset(ArgGlobalOffset) {}                                     \
  };                                                                           \
                                                                               \
  template <typename LambdaTy>                                                 \
  auto makeWrapper_##TAG(LambdaTy F, const sycl::range<DIMS> &LocalSize,       \
                         const sycl::range<DIMS> &GlobalSize,                  \
                         const sycl::id<DIMS> &GlobalOffset) {                 \
    std::unique_ptr<LambdaWrapper_##TAG> Wrapper =                             \
        std::make_unique<LambdaWrapper_##TAG>(LambdaWrapper_##TAG(             \
            LambdaFunction_##TAG(F), LocalSize, GlobalSize, GlobalOffset));    \
    return Wrapper;                                                            \
  }

#define _COMMA_ ,

LAMBDA_WRAPPER_TMPL(sycl::id<1>, ID_1DIM, 1)
LAMBDA_WRAPPER_TMPL(sycl::id<2>, ID_2DIM, 2)
LAMBDA_WRAPPER_TMPL(sycl::id<3>, ID_3DIM, 3)
LAMBDA_WRAPPER_TMPL(sycl::item<1 _COMMA_ false>, ITEM_1DIM, 1)
LAMBDA_WRAPPER_TMPL(sycl::item<2 _COMMA_ false>, ITEM_2DIM, 2)
LAMBDA_WRAPPER_TMPL(sycl::item<3 _COMMA_ false>, ITEM_3DIM, 3)
LAMBDA_WRAPPER_TMPL(sycl::item<1 _COMMA_ true>, ITEM_OFFSET_1DIM, 1)
LAMBDA_WRAPPER_TMPL(sycl::item<2 _COMMA_ true>, ITEM_OFFSET_2DIM, 2)
LAMBDA_WRAPPER_TMPL(sycl::item<3 _COMMA_ true>, ITEM_OFFSET_3DIM, 3)
LAMBDA_WRAPPER_TMPL(sycl::nd_item<1>, NDITEM_1DIM, 1)
LAMBDA_WRAPPER_TMPL(sycl::nd_item<2>, NDITEM_2DIM, 2)
LAMBDA_WRAPPER_TMPL(sycl::nd_item<3>, NDITEM_3DIM, 3)

#undef _COMMA_
#undef LAMBDA_WRAPPER_TMPL

extern "C" inline void invokeLambda_ID_1DIM(void *Wrapper) {
  auto *LambdaWrapper = reinterpret_cast<LambdaWrapper_ID_1DIM *>(Wrapper);

  PLACEHOLDER_UNUSED(LambdaWrapper);
  // TODO : construct cl::sycl::id<1> instance using thread ID info
  // retrieved from CM and call Lambda function
  // LambdaWrapper->Func(id_1dim);
}

extern "C" inline void invokeLambda_ID_2DIM(void *Wrapper) {
  auto *LambdaWrapper = reinterpret_cast<LambdaWrapper_ID_2DIM *>(Wrapper);

  PLACEHOLDER_UNUSED(LambdaWrapper);
  // TODO : construct cl::sycl::id<2> instance using thread ID info
  // retrieved from CM and call Lambda function
  // LambdaWrapper->Func(id_2dim);
}

extern "C" inline void invokeLambda_ID_3DIM(void *Wrapper) {
  auto *LambdaWrapper = reinterpret_cast<LambdaWrapper_ID_3DIM *>(Wrapper);

  PLACEHOLDER_UNUSED(LambdaWrapper);
  // TODO : construct cl::sycl::id<3> instance using thread ID info
  // retrieved from CM and call Lambda function
  // LambdaWrapper->Func(id_3dim);
}

extern "C" inline void invokeLambda_ITEM_1DIM(void *Wrapper) {
  auto *LambdaWrapper = reinterpret_cast<LambdaWrapper_ITEM_1DIM *>(Wrapper);

  PLACEHOLDER_UNUSED(LambdaWrapper);
  // TODO : construct cl::sycl::item<1, false> instance using thread
  // ID info retrieved from CM and call Lambda function
  // LambdaWrapper->Func(item_1dim);
}

extern "C" inline void invokeLambda_ITEM_2DIM(void *Wrapper) {
  auto *LambdaWrapper = reinterpret_cast<LambdaWrapper_ITEM_2DIM *>(Wrapper);

  PLACEHOLDER_UNUSED(LambdaWrapper);
  // TODO : construct cl::sycl::item<2, false> instance using thread
  // ID info retrieved from CM and call Lambda function
  // LambdaWrapper->Func(item_2dim);
}

extern "C" inline void invokeLambda_ITEM_3DIM(void *Wrapper) {
  auto *LambdaWrapper = reinterpret_cast<LambdaWrapper_ITEM_3DIM *>(Wrapper);

  PLACEHOLDER_UNUSED(LambdaWrapper);
  // TODO : construct cl::sycl::item<3, false> instance using thread
  // ID info retrieved from CM and call Lambda function
  // LambdaWrapper->Func(item_3dim);
}

extern "C" inline void invokeLambda_ITEM_OFFSET_1DIM(void *Wrapper) {
  auto *LambdaWrapper =
      reinterpret_cast<LambdaWrapper_ITEM_OFFSET_1DIM *>(Wrapper);

  PLACEHOLDER_UNUSED(LambdaWrapper);
  // TODO : construct cl::sycl::item<1, true> instance using thread
  // ID info retrieved from CM with GlobalOffset info and call Lambda
  // function
  // LambdaWrapper->Func(item_offset_1dim);
}

extern "C" inline void invokeLambda_ITEM_OFFSET_2DIM(void *Wrapper) {
  auto *LambdaWrapper =
      reinterpret_cast<LambdaWrapper_ITEM_OFFSET_2DIM *>(Wrapper);

  PLACEHOLDER_UNUSED(LambdaWrapper);
  // TODO : construct cl::sycl::item<2, true> instance using thread
  // ID info retrieved from CM with GlobalOffset info and call Lambda
  // function
  // LambdaWrapper->Func(item_offset_2dim);
}

extern "C" inline void invokeLambda_ITEM_OFFSET_3DIM(void *Wrapper) {
  auto *LambdaWrapper =
      reinterpret_cast<LambdaWrapper_ITEM_OFFSET_3DIM *>(Wrapper);

  PLACEHOLDER_UNUSED(LambdaWrapper);
  // TODO : construct cl::sycl::item<3, true> instance using thread
  // ID info retrieved from CM with GlobalOffset info and call Lambda
  // function
  // LambdaWrapper->Func(item_offset_3dim);
}

extern "C" inline void invokeLambda_NDITEM_1DIM(void *Wrapper) {
  auto *LambdaWrapper = reinterpret_cast<LambdaWrapper_NDITEM_1DIM *>(Wrapper);

  PLACEHOLDER_UNUSED(LambdaWrapper);
  // TODO : construct cl::sycl::nd_item<1> instance using thread ID
  // info retrieved from CM with GlobalOffset/GlobalSize/LocalSize
  // info and call Lambda function
  // LambdaWrapper->Func(nd_item_1dim);
}

extern "C" inline void invokeLambda_NDITEM_2DIM(void *Wrapper) {
  auto *LambdaWrapper = reinterpret_cast<LambdaWrapper_NDITEM_2DIM *>(Wrapper);

  PLACEHOLDER_UNUSED(LambdaWrapper);
  // TODO : construct cl::sycl::nd_item<2> instance using thread ID
  // info retrieved from CM with GlobalOffset/GlobalSize/LocalSize
  // info and call Lambda function
  // LambdaWrapper->Func(nd_item_2dim);
}

extern "C" inline void invokeLambda_NDITEM_3DIM(void *Wrapper) {
  auto *LambdaWrapper = reinterpret_cast<LambdaWrapper_NDITEM_3DIM *>(Wrapper);

  PLACEHOLDER_UNUSED(LambdaWrapper);
  // TODO : construct cl::sycl::nd_item<3> instance using thread ID
  // info retrieved from CM with GlobalOffset/GlobalSize/LocalSize
  // info and call Lambda function
  // LambdaWrapper->Func(nd_item_3dim);
}

// libCMBatch class defines interface for lauching kernels with
// software multi-threads
template <class KernelType, class KernelArgType, int DIMS> class libCMBatch {
private:
  // Kernel function
  KernelType MKernel;

  // Space-dimension info
  std::vector<uint32_t> GroupDim;
  std::vector<uint32_t> SpaceDim;

  // Number of threads for parallelization
  const uint32_t hwThreads = (uint32_t)std::thread::hardware_concurrency();

  using IDBuilder = sycl::detail::Builder;
  const sycl::id<DIMS> UnusedID =
      sycl::detail::InitializedVal<DIMS, sycl::id>::template get<0>();
  const sycl::range<DIMS> UnusedRange =
      sycl::detail::InitializedVal<DIMS, sycl::range>::template get<0>();

public:
  libCMBatch(KernelType Kernel)
      : MKernel(Kernel), GroupDim{1, 1, 1}, SpaceDim{1, 1, 1} {
    assert(MKernel != nullptr);
  }

  // ID_1DIM
  template <class ArgT = KernelArgType>
  typename std::enable_if<(DIMS == 1) &&
                          (std::is_same<ArgT, sycl::id<1>>::value)>::type
  runIterationSpace(const sycl::range<1> &Range) {
    auto WrappedLambda_ID_1DIM =
        makeWrapper_ID_1DIM(MKernel, UnusedRange, UnusedRange, UnusedID);

    SpaceDim[0] = (uint32_t)Range[0];

    PLACEHOLDER_UNUSED(WrappedLambda_ID_1DIM);
    // TODO : Invoke invokeLambda_ID_1DIM through CM's multi-threaded
    // kernel launching with WrappedLambda_ID_1DIM and dimension info
  }

  // ID_2DIM
  template <class ArgT = KernelArgType>
  typename std::enable_if<(DIMS == 2) &&
                          (std::is_same<ArgT, sycl::id<2>>::value)>::type
  runIterationSpace(const sycl::range<2> &Range) {
    auto WrappedLambda_ID_2DIM =
        makeWrapper_ID_2DIM(MKernel, UnusedRange, UnusedRange, UnusedID);

    SpaceDim[0] = (uint32_t)Range[0];
    SpaceDim[1] = (uint32_t)Range[1];

    PLACEHOLDER_UNUSED(WrappedLambda_ID_2DIM);
    // TODO : Invoke invokeLambda_ID_2DIM through CM's multi-threaded
    // kernel launching with WrappedLambda_ID_2DIM and dimension info
  }

  // ID_3DIM
  template <class ArgT = KernelArgType>
  typename std::enable_if<(DIMS == 3) &&
                          (std::is_same<ArgT, sycl::id<3>>::value)>::type
  runIterationSpace(const sycl::range<3> &Range) {
    auto WrappedLambda_ID_3DIM =
        makeWrapper_ID_3DIM(MKernel, UnusedRange, UnusedRange, UnusedID);

    SpaceDim[0] = (uint32_t)Range[0];
    SpaceDim[1] = (uint32_t)Range[1];
    SpaceDim[2] = (uint32_t)Range[2];

    PLACEHOLDER_UNUSED(WrappedLambda_ID_3DIM);
    // TODO : Invoke invokeLambda_ID_3DIM through CM's multi-threaded
    // kernel launching with WrappedLambda_ID_3DIM and dimension info
  }

  // Item w/o offset
  template <class ArgT = KernelArgType>
  typename std::enable_if<
      (DIMS == 1) &&
      (std::is_same<ArgT, sycl::item<1, /*Offset=*/false>>::value)>::type
  runIterationSpace(const sycl::range<1> &Range) {
    auto WrappedLambda_ITEM_1DIM =
        makeWrapper_ITEM_1DIM(MKernel, UnusedRange, UnusedRange, UnusedID);

    SpaceDim[0] = (uint32_t)Range[0];

    PLACEHOLDER_UNUSED(WrappedLambda_ITEM_1DIM);
    // TODO : Invoke invokeLambda_ITEM_1DIM through CM's multi-threaded
    // kernel launching with WrappedLambda_ITEM_1DIM and dimension info
  }

  template <class ArgT = KernelArgType>
  typename std::enable_if<
      (DIMS == 2) &&
      (std::is_same<ArgT, sycl::item<2, /*Offset=*/false>>::value)>::type
  runIterationSpace(const sycl::range<2> &Range) {
    auto WrappedLambda_ITEM_2DIM =
        makeWrapper_ITEM_2DIM(MKernel, UnusedRange, UnusedRange, UnusedID);

    SpaceDim[0] = (uint32_t)Range[0];
    SpaceDim[1] = (uint32_t)Range[1];

    PLACEHOLDER_UNUSED(WrappedLambda_ITEM_2DIM);
    // TODO : Invoke invokeLambda_ITEM_2DIM through CM's multi-threaded
    // kernel launching with WrappedLambda_ITEM_2DIM and dimension info
  }

  template <class ArgT = KernelArgType>
  typename std::enable_if<
      (DIMS == 3) &&
      (std::is_same<ArgT, sycl::item<3, /*Offset=*/false>>::value)>::type
  runIterationSpace(const sycl::range<3> &Range) {
    auto WrappedLambda_ITEM_3DIM =
        makeWrapper_ITEM_3DIM(MKernel, UnusedRange, UnusedRange, UnusedID);

    SpaceDim[0] = (uint32_t)Range[0];
    SpaceDim[1] = (uint32_t)Range[1];
    SpaceDim[2] = (uint32_t)Range[2];

    PLACEHOLDER_UNUSED(WrappedLambda_ITEM_3DIM);
    // TODO : Invoke invokeLambda_ITEM_3DIM through CM's multi-threaded
    // kernel launching with WrappedLambda_ITEM_3DIM and dimension info
  }

  // Item w/ offset
  template <class ArgT = KernelArgType>
  typename std::enable_if<
      (DIMS == 1) &&
      (std::is_same<ArgT, sycl::item<1, /*Offset=*/true>>::value)>::type
  runIterationSpace(const sycl::range<1> &Range, const sycl::id<1> &Offset) {
    auto WrappedLambda_ITEM_OFFSET_1DIM =
        makeWrapper_ITEM_OFFSET_1DIM(MKernel, UnusedRange, UnusedRange, Offset);

    SpaceDim[0] = (uint32_t)Range[0];

    PLACEHOLDER_UNUSED(WrappedLambda_ITEM_OFFSET_1DIM);
    // TODO : Invoke invokeLambda_ITEM_OFFSET_1DIM through CM's multi-threaded
    // kernel launching with WrappedLambda_ITEM_OFFSET_1DIM and dimension info
  }

  template <class ArgT = KernelArgType>
  typename std::enable_if<
      (DIMS == 2) &&
      (std::is_same<ArgT, sycl::item<2, /*Offset=*/true>>::value)>::type
  runIterationSpace(const sycl::range<2> &Range, const sycl::id<2> &Offset) {
    auto WrappedLambda_ITEM_OFFSET_2DIM =
        makeWrapper_ITEM_OFFSET_2DIM(MKernel, UnusedRange, UnusedRange, Offset);

    SpaceDim[0] = (uint32_t)Range[0];
    SpaceDim[1] = (uint32_t)Range[1];

    PLACEHOLDER_UNUSED(WrappedLambda_ITEM_OFFSET_2DIM);
    // TODO : Invoke invokeLambda_ITEM_OFFSET_2DIM through CM's multi-threaded
    // kernel launching with WrappedLambda_ITEM_OFFSET_2DIM and dimension info
  }

  template <class ArgT = KernelArgType>
  typename std::enable_if<
      (DIMS == 3) &&
      (std::is_same<ArgT, sycl::item<3, /*Offset=*/true>>::value)>::type
  runIterationSpace(const sycl::range<3> &Range, const sycl::id<3> &Offset) {
    auto WrappedLambda_ITEM_OFFSET_3DIM =
        makeWrapper_ITEM_OFFSET_3DIM(MKernel, UnusedRange, UnusedRange, Offset);

    SpaceDim[0] = (uint32_t)Range[0];
    SpaceDim[1] = (uint32_t)Range[1];
    SpaceDim[2] = (uint32_t)Range[2];

    PLACEHOLDER_UNUSED(WrappedLambda_ITEM_OFFSET_3DIM);
    // TODO : Invoke invokeLambda_ITEM_OFFSET_3DIM through CM's multi-threaded
    // kernel launching with WrappedLambda_ITEM_OFFSET_3DIM and dimension info
  }

  // NDItem_1DIM
  template <class ArgT = KernelArgType>
  typename std::enable_if<(DIMS == 1) &&
                          (std::is_same<ArgT, sycl::nd_item<1>>::value)>::type
  runIterationSpace(const sycl::range<1> &LocalSize,
                    const sycl::range<1> &GlobalSize,
                    const sycl::id<1> &GlobalOffset) {
    auto WrappedLambda_NDITEM_1DIM =
        makeWrapper_NDITEM_1DIM(MKernel, LocalSize, GlobalSize, GlobalOffset);

    SpaceDim[0] = (uint32_t)LocalSize[0];

    GroupDim[0] = (uint32_t)(GlobalSize[0] / LocalSize[0]);

    PLACEHOLDER_UNUSED(WrappedLambda_NDITEM_1DIM);
    // TODO : Invoke invokeLambda_NDITEM_1DIM through CM's multi-threaded
    // kernel launching with WrappedLambda_NDITEM_1DIM and dimension info
  }

  // NDItem_2DIM
  template <class ArgT = KernelArgType>
  typename std::enable_if<(DIMS == 2) &&
                          (std::is_same<ArgT, sycl::nd_item<2>>::value)>::type
  runIterationSpace(const sycl::range<2> &LocalSize,
                    const sycl::range<2> &GlobalSize,
                    const sycl::id<2> &GlobalOffset) {
    auto WrappedLambda_NDITEM_2DIM =
        makeWrapper_NDITEM_2DIM(MKernel, LocalSize, GlobalSize, GlobalOffset);

    SpaceDim[0] = (uint32_t)LocalSize[0];
    SpaceDim[1] = (uint32_t)LocalSize[1];

    GroupDim[0] = (uint32_t)(GlobalSize[0] / LocalSize[0]);
    GroupDim[1] = (uint32_t)(GlobalSize[1] / LocalSize[1]);

    PLACEHOLDER_UNUSED(WrappedLambda_NDITEM_2DIM);
    // TODO : Invoke invokeLambda_NDITEM_2DIM through CM's multi-threaded
    // kernel launching with WrappedLambda_NDITEM_2DIM and dimension info
  }

  // NDItem_3DIM
  template <class ArgT = KernelArgType>
  typename std::enable_if<(DIMS == 3) &&
                          (std::is_same<ArgT, sycl::nd_item<3>>::value)>::type
  runIterationSpace(const sycl::range<3> &LocalSize,
                    const sycl::range<3> &GlobalSize,
                    const sycl::id<3> &GlobalOffset) {
    auto WrappedLambda_NDITEM_3DIM =
        makeWrapper_NDITEM_3DIM(MKernel, LocalSize, GlobalSize, GlobalOffset);

    SpaceDim[0] = (uint32_t)LocalSize[0];
    SpaceDim[1] = (uint32_t)LocalSize[1];
    SpaceDim[2] = (uint32_t)LocalSize[2];

    GroupDim[0] = (uint32_t)(GlobalSize[0] / LocalSize[0]);
    GroupDim[1] = (uint32_t)(GlobalSize[1] / LocalSize[1]);
    GroupDim[2] = (uint32_t)(GlobalSize[2] / LocalSize[2]);

    PLACEHOLDER_UNUSED(WrappedLambda_NDITEM_3DIM);
    // TODO : Invoke invokeLambda_NDITEM_3DIM through CM's multi-threaded
    // kernel launching with WrappedLambda_NDITEM_3DIM and dimension info
  }
};

// Intrinsics
sycl::detail::ESIMDDeviceInterface::ESIMDDeviceInterface() {
  reserved = nullptr;
  version = ESIMDEmuPluginInterfaceVersion;

  /// TODO : Fill *_ptr fields with function pointers from CM
  /// functions prefixed with 'cm_support'

  cm_barrier_ptr = nullptr;  /* cm_support::barrier; */
  cm_sbarrier_ptr = nullptr; /* cm_support::split_barrier; */
  cm_fence_ptr = nullptr;    /* cm_support::fence; */

  sycl_get_surface_base_addr_ptr =
      nullptr;                    /* cm_support::get_surface_base_addr; */
  __cm_emu_get_slm_ptr = nullptr; /* cm_support::get_slm_base; */
  cm_slm_init_ptr = nullptr;      /* cm_support::init_slm; */
}

extern "C" {

#define DIE_NO_IMPLEMENTATION                                                  \
  std::cerr << "Not Implemented : " << __FUNCTION__                            \
            << " - File : " << __FILE__;                                       \
  std::cerr << " / Line : " << __LINE__ << std::endl;                          \
  die("Terminated")

#define DIE_NO_SUPPORT                                                         \
  std::cerr << "Not Supported : " << __FUNCTION__ << " - File : " << __FILE__; \
  std::cerr << " / Line : " << __LINE__ << std::endl;                          \
  die("Terminated")

#define CONTINUE_NO_IMPLEMENTATION                                             \
  std::cerr << "Warning : Not Implemented : " << __FUNCTION__                  \
            << " - File : " << __FILE__;                                       \
  std::cerr << " / Line : " << __LINE__ << std::endl;

pi_result piPlatformsGet(pi_uint32 NumEntries, pi_platform *Platforms,
                         pi_uint32 *NumPlatforms) {
  (void)NumEntries;
  (void)Platforms;
  (void)NumPlatforms;
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piPlatformGetInfo(pi_platform Platform, pi_platform_info ParamName,
                            size_t ParamValueSize, void *ParamValue,
                            size_t *ParamValueSizeRet) {
  (void)Platform;
  (void)ParamName;
  (void)ParamValueSize;
  (void)ParamValue;
  (void)ParamValueSizeRet;
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextPlatformGetNativeHandle(pi_platform, pi_native_handle *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextPlatformCreateWithNativeHandle(pi_native_handle, pi_platform *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piDevicesGet(pi_platform Platform, pi_device_type DeviceType,
                       pi_uint32 NumEntries, pi_device *Devices,
                       pi_uint32 *NumDevices) {
  (void)Platform;
  (void)DeviceType;
  (void)NumEntries;
  (void)Devices;
  (void)NumDevices;
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piDeviceRetain(pi_device Device) {
  (void)Device;
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piDeviceRelease(pi_device) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piDeviceGetInfo(pi_device Device, pi_device_info ParamName,
                          size_t ParamValueSize, void *ParamValue,
                          size_t *ParamValueSizeRet) {
  (void)Device;
  (void)ParamName;
  (void)ParamValueSize;
  (void)ParamValue;
  (void)ParamValueSizeRet;
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piDevicePartition(pi_device, const pi_device_partition_property *,
                            pi_uint32, pi_device *, pi_uint32 *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextDeviceGetNativeHandle(pi_device, pi_native_handle *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextDeviceCreateWithNativeHandle(pi_native_handle, pi_platform,
                                            pi_device *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piContextCreate(const pi_context_properties *Properties,
                          pi_uint32 NumDevices, const pi_device *Devices,
                          void (*PFnNotify)(const char *ErrInfo,
                                            const void *PrivateInfo, size_t CB,
                                            void *UserData),
                          void *UserData, pi_context *RetContext) {
  (void)Properties;
  (void)NumDevices;
  (void)Devices;
  (void)PFnNotify;
  (void)UserData;
  (void)RetContext;
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piContextGetInfo(pi_context, pi_context_info, size_t, void *,
                           size_t *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextContextSetExtendedDeleter(pi_context,
                                         pi_context_extended_deleter, void *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextContextGetNativeHandle(pi_context, pi_native_handle *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextContextCreateWithNativeHandle(pi_native_handle, pi_uint32,
                                             const pi_device *, bool,
                                             pi_context *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piContextRetain(pi_context) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piContextRelease(pi_context Context) {
  (void)Context;
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piQueueCreate(pi_context Context, pi_device Device,
                        pi_queue_properties Properties, pi_queue *Queue) {
  (void)Context;
  (void)Device;
  (void)Properties;
  (void)Queue;
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piQueueGetInfo(pi_queue, pi_queue_info, size_t, void *, size_t *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piQueueRetain(pi_queue) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piQueueRelease(pi_queue Queue) {
  (void)Queue;
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piQueueFinish(pi_queue) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextQueueGetNativeHandle(pi_queue, pi_native_handle *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextQueueCreateWithNativeHandle(pi_native_handle, pi_context,
                                           pi_queue *, bool) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piMemBufferCreate(pi_context Context, pi_mem_flags Flags, size_t Size,
                            void *HostPtr, pi_mem *RetMem,
                            const pi_mem_properties *properties) {
  (void)Context;
  (void)Flags;
  (void)Size;
  (void)HostPtr;
  (void)RetMem;
  (void)properties;
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piMemGetInfo(pi_mem, cl_mem_info, size_t, void *, size_t *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piMemRetain(pi_mem) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piMemRelease(pi_mem Mem) {
  (void)Mem;
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piMemImageCreate(pi_context Context, pi_mem_flags Flags,
                           const pi_image_format *ImageFormat,
                           const pi_image_desc *ImageDesc, void *HostPtr,
                           pi_mem *RetImage) {
  (void)Context;
  (void)Flags;
  (void)ImageFormat;
  (void)ImageDesc;
  (void)HostPtr;
  (void)RetImage;
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextMemGetNativeHandle(pi_mem, pi_native_handle *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextMemCreateWithNativeHandle(pi_native_handle, pi_mem *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piProgramCreate(pi_context, const void *, size_t, pi_program *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piProgramCreateWithBinary(pi_context, pi_uint32, const pi_device *,
                                    const size_t *, const unsigned char **,
                                    size_t, const pi_device_binary_property *,
                                    pi_int32 *, pi_program *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piclProgramCreateWithBinary(pi_context, pi_uint32, const pi_device *,
                                      const size_t *, const unsigned char **,
                                      pi_int32 *, pi_program *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piclProgramCreateWithSource(pi_context, pi_uint32, const char **,
                                      const size_t *, pi_program *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piProgramGetInfo(pi_program, pi_program_info, size_t, void *,
                           size_t *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piProgramLink(pi_context, pi_uint32, const pi_device *, const char *,
                        pi_uint32, const pi_program *,
                        void (*)(pi_program, void *), void *, pi_program *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piProgramCompile(pi_program, pi_uint32, const pi_device *,
                           const char *, pi_uint32, const pi_program *,
                           const char **, void (*)(pi_program, void *),
                           void *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piProgramBuild(pi_program, pi_uint32, const pi_device *, const char *,
                         void (*)(pi_program, void *), void *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piProgramGetBuildInfo(pi_program, pi_device, cl_program_build_info,
                                size_t, void *, size_t *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piProgramRetain(pi_program) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piProgramRelease(pi_program) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextProgramGetNativeHandle(pi_program, pi_native_handle *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextProgramCreateWithNativeHandle(pi_native_handle, pi_context,
                                             pi_program *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piKernelCreate(pi_program, const char *, pi_kernel *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piKernelSetArg(pi_kernel, pi_uint32, size_t, const void *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextKernelSetArgMemObj(pi_kernel, pi_uint32, const pi_mem *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

// Special version of piKernelSetArg to accept pi_sampler.
pi_result piextKernelSetArgSampler(pi_kernel, pi_uint32, const pi_sampler *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piKernelGetInfo(pi_kernel, pi_kernel_info, size_t, void *, size_t *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piKernelGetGroupInfo(pi_kernel, pi_device, pi_kernel_group_info,
                               size_t, void *, size_t *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result
piKernelGetSubGroupInfo(pi_kernel, pi_device,
                        pi_kernel_sub_group_info, // TODO: untie from OpenCL
                        size_t, const void *, size_t, void *, size_t *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piKernelRetain(pi_kernel) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piKernelRelease(pi_kernel) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEventCreate(pi_context, pi_event *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEventGetInfo(pi_event, pi_event_info, size_t, void *, size_t *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEventGetProfilingInfo(pi_event Event, pi_profiling_info ParamName,
                                  size_t ParamValueSize, void *ParamValue,
                                  size_t *ParamValueSizeRet) {
  (void)Event;
  (void)ParamName;
  (void)ParamValueSize;
  (void)ParamValue;
  (void)ParamValueSizeRet;
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEventsWait(pi_uint32 NumEvents, const pi_event *EventList) {
  (void)NumEvents;
  (void)EventList;
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEventSetCallback(pi_event, pi_int32,
                             void (*)(pi_event, pi_int32, void *), void *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEventSetStatus(pi_event, pi_int32) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEventRetain(pi_event) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEventRelease(pi_event Event) {
  (void)Event;
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextEventGetNativeHandle(pi_event, pi_native_handle *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextEventCreateWithNativeHandle(pi_native_handle, pi_context, bool,
                                           pi_event *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piSamplerCreate(pi_context, const pi_sampler_properties *,
                          pi_sampler *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piSamplerGetInfo(pi_sampler, pi_sampler_info, size_t, void *,
                           size_t *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piSamplerRetain(pi_sampler) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piSamplerRelease(pi_sampler) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEnqueueEventsWait(pi_queue, pi_uint32, const pi_event *,
                              pi_event *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEnqueueEventsWaitWithBarrier(pi_queue, pi_uint32, const pi_event *,
                                         pi_event *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEnqueueMemBufferRead(pi_queue Queue, pi_mem Src,
                                 pi_bool BlockingRead, size_t Offset,
                                 size_t Size, void *Dst,
                                 pi_uint32 NumEventsInWaitList,
                                 const pi_event *EventWaitList,
                                 pi_event *Event) {
  (void)Queue;
  (void)Src;
  (void)BlockingRead;
  (void)Offset;
  (void)Size;
  (void)Dst;
  (void)NumEventsInWaitList;
  (void)EventWaitList;
  (void)Event;
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEnqueueMemBufferReadRect(pi_queue, pi_mem, pi_bool,
                                     pi_buff_rect_offset, pi_buff_rect_offset,
                                     pi_buff_rect_region, size_t, size_t,
                                     size_t, size_t, void *, pi_uint32,
                                     const pi_event *, pi_event *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEnqueueMemBufferWrite(pi_queue, pi_mem, pi_bool, size_t, size_t,
                                  const void *, pi_uint32, const pi_event *,
                                  pi_event *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEnqueueMemBufferWriteRect(pi_queue, pi_mem, pi_bool,
                                      pi_buff_rect_offset, pi_buff_rect_offset,
                                      pi_buff_rect_region, size_t, size_t,
                                      size_t, size_t, const void *, pi_uint32,
                                      const pi_event *, pi_event *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEnqueueMemBufferCopy(pi_queue, pi_mem, pi_mem, size_t, size_t,
                                 size_t, pi_uint32, const pi_event *,
                                 pi_event *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEnqueueMemBufferCopyRect(pi_queue, pi_mem, pi_mem,
                                     pi_buff_rect_offset, pi_buff_rect_offset,
                                     pi_buff_rect_region, size_t, size_t,
                                     size_t, size_t, pi_uint32,
                                     const pi_event *, pi_event *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEnqueueMemBufferFill(pi_queue, pi_mem, const void *, size_t, size_t,
                                 size_t, pi_uint32, const pi_event *,
                                 pi_event *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEnqueueMemBufferMap(pi_queue, pi_mem, pi_bool, pi_map_flags, size_t,
                                size_t, pi_uint32, const pi_event *, pi_event *,
                                void **) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEnqueueMemUnmap(pi_queue, pi_mem, void *, pi_uint32,
                            const pi_event *, pi_event *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piMemImageGetInfo(pi_mem, pi_image_info, size_t, void *, size_t *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEnqueueMemImageRead(pi_queue command_queue, pi_mem image,
                                pi_bool blocking_read, pi_image_offset origin,
                                pi_image_region region, size_t row_pitch,
                                size_t slice_pitch, void *ptr,
                                pi_uint32 num_events_in_wait_list,
                                const pi_event *event_wait_list,
                                pi_event *event) {
  (void)command_queue;
  (void)image;
  (void)blocking_read;
  (void)origin;
  (void)region;
  (void)row_pitch;
  (void)slice_pitch;
  (void)ptr;
  (void)num_events_in_wait_list;
  (void)event_wait_list;
  (void)event;
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEnqueueMemImageWrite(pi_queue, pi_mem, pi_bool, pi_image_offset,
                                 pi_image_region, size_t, size_t, const void *,
                                 pi_uint32, const pi_event *, pi_event *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEnqueueMemImageCopy(pi_queue, pi_mem, pi_mem, pi_image_offset,
                                pi_image_offset, pi_image_region, pi_uint32,
                                const pi_event *, pi_event *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEnqueueMemImageFill(pi_queue, pi_mem, const void *, const size_t *,
                                const size_t *, pi_uint32, const pi_event *,
                                pi_event *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piMemBufferPartition(pi_mem, pi_mem_flags, pi_buffer_create_type,
                               void *, pi_mem *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result
piEnqueueKernelLaunch(pi_queue Queue, pi_kernel Kernel, pi_uint32 WorkDim,
                      const size_t *GlobalWorkOffset,
                      const size_t *GlobalWorkSize, const size_t *LocalWorkSize,
                      pi_uint32 NumEventsInWaitList,
                      const pi_event *EventWaitList, pi_event *Event) {
  (void)Queue;
  (void)Kernel;
  (void)WorkDim;
  (void)GlobalWorkOffset;
  (void)GlobalWorkSize;
  (void)LocalWorkSize;
  (void)NumEventsInWaitList;
  (void)EventWaitList;
  (void)Event;
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextKernelCreateWithNativeHandle(pi_native_handle, pi_context, bool,
                                            pi_kernel *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextKernelGetNativeHandle(pi_kernel, pi_native_handle *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEnqueueNativeKernel(pi_queue, void (*)(void *), void *, size_t,
                                pi_uint32, const pi_mem *, const void **,
                                pi_uint32, const pi_event *, pi_event *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextGetDeviceFunctionPointer(pi_device, pi_program, const char *,
                                        pi_uint64 *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextUSMHostAlloc(void **, pi_context, pi_usm_mem_properties *,
                            size_t, pi_uint32) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextUSMDeviceAlloc(void **, pi_context, pi_device,
                              pi_usm_mem_properties *, size_t, pi_uint32) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextUSMSharedAlloc(void **ResultPtr, pi_context Context,
                              pi_device Device,
                              pi_usm_mem_properties *Properties, size_t Size,
                              pi_uint32 Alignment) {
  (void)ResultPtr;
  (void)Context;
  (void)Device;
  (void)Properties;
  (void)Size;
  (void)Alignment;
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextUSMFree(pi_context Context, void *Ptr) {
  (void)Context;
  (void)Ptr;
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextKernelSetArgPointer(pi_kernel, pi_uint32, size_t, const void *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextUSMEnqueueMemset(pi_queue, void *, pi_int32, size_t, pi_uint32,
                                const pi_event *, pi_event *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextUSMEnqueueMemcpy(pi_queue, pi_bool, void *, const void *, size_t,
                                pi_uint32, const pi_event *, pi_event *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextUSMEnqueueMemAdvise(pi_queue, const void *, size_t,
                                   pi_mem_advice, pi_event *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextUSMGetMemAllocInfo(pi_context, const void *, pi_mem_info, size_t,
                                  void *, size_t *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piKernelSetExecInfo(pi_kernel, pi_kernel_exec_info, size_t,
                              const void *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextProgramSetSpecializationConstant(pi_program, pi_uint32, size_t,
                                                const void *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextDeviceSelectBinary(pi_device, pi_device_binary *, pi_uint32,
                                  pi_uint32 *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextUSMEnqueuePrefetch(pi_queue, const void *, size_t,
                                  pi_usm_migration_flags, pi_uint32,
                                  const pi_event *, pi_event *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextPluginGetOpaqueData(void *, void **opaque_data_return) {
  *opaque_data_return = reinterpret_cast<void *>(PiESimdDeviceAccess);
  return PI_SUCCESS;
}

pi_result piTearDown(void *) {
  delete reinterpret_cast<sycl::detail::ESIMDEmuPluginOpaqueData *>(
      PiESimdDeviceAccess->data);
  delete PiESimdDeviceAccess;
  return PI_SUCCESS;
}

pi_result piPluginInit(pi_plugin *PluginInit) {
  assert(PluginInit);
  size_t PluginVersionSize = sizeof(PluginInit->PluginVersion);
  assert(strlen(_PI_H_VERSION_STRING) < PluginVersionSize);
  strncpy(PluginInit->PluginVersion, _PI_H_VERSION_STRING, PluginVersionSize);

  PiESimdDeviceAccess = new sycl::detail::ESIMDEmuPluginOpaqueData();
  // 'version' to be compared with 'ESIMD_CPU_DEVICE_REQUIRED_VER' defined in
  // device interface file
  PiESimdDeviceAccess->version = ESIMDEmuPluginDataVersion;
  PiESimdDeviceAccess->data =
      reinterpret_cast<void *>(new sycl::detail::ESIMDDeviceInterface());

#define _PI_API(api)                                                           \
  (PluginInit->PiFunctionTable).api = (decltype(&::api))(&api);
#include <CL/sycl/detail/pi.def>

  return PI_SUCCESS;
}

} // extern C
