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

#include <esimdcpu_support.h>

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include "pi_esimd_cpu.hpp"

namespace {

template <typename T, typename Assign>
pi_result getInfoImpl(size_t param_value_size, void *param_value,
                      size_t *param_value_size_ret, T value, size_t value_size,
                      Assign &&assign_func) {

  if (param_value != nullptr) {

    if (param_value_size < value_size) {
      return PI_INVALID_VALUE;
    }

    assign_func(param_value, value, value_size);
  }

  if (param_value_size_ret != nullptr) {
    *param_value_size_ret = value_size;
  }

  return PI_SUCCESS;
}

template <typename T>
pi_result getInfo(size_t param_value_size, void *param_value,
                  size_t *param_value_size_ret, T value) {

  auto assignment = [](void *param_value, T value, size_t value_size) {
    *static_cast<T *>(param_value) = value;
  };

  return getInfoImpl(param_value_size, param_value, param_value_size_ret, value,
                     sizeof(T), assignment);
}

template <typename T>
pi_result getInfoArray(size_t array_length, size_t param_value_size,
                       void *param_value, size_t *param_value_size_ret,
                       T *value) {
  return getInfoImpl(param_value_size, param_value, param_value_size_ret, value,
                     array_length * sizeof(T), memcpy);
}

template <>
pi_result getInfo<const char *>(size_t param_value_size, void *param_value,
                                size_t *param_value_size_ret,
                                const char *value) {
  return getInfoArray(strlen(value) + 1, param_value_size, param_value,
                      param_value_size_ret, value);
}

class ReturnHelper {
public:
  ReturnHelper(size_t param_value_size, void *param_value,
               size_t *param_value_size_ret)
      : param_value_size(param_value_size), param_value(param_value),
        param_value_size_ret(param_value_size_ret) {}

  template <class T> pi_result operator()(const T &t) {
    return getInfo(param_value_size, param_value, param_value_size_ret, t);
  }

private:
  size_t param_value_size;
  void *param_value;
  size_t *param_value_size_ret;
};

} // anonymous namespace

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

extern "C" {

inline void InvokeLambda_ID_1DIM(void *Wrapper) {
  auto *LambdaWrapper = reinterpret_cast<LambdaWrapper_ID_1DIM *>(Wrapper);
  cl::sycl::id<1> Id1Dim(cm_support::get_thread_idx(0));
  LambdaWrapper->Func(Id1Dim);
}

inline void InvokeLambda_ID_2DIM(void *Wrapper) {
  auto *LambdaWrapper = reinterpret_cast<LambdaWrapper_ID_2DIM *>(Wrapper);
  cl::sycl::id<2> Id2Dim(cm_support::get_thread_idx(0),
                         cm_support::get_thread_idx(1));
  LambdaWrapper->Func(Id2Dim);
}

inline void InvokeLambda_ID_3DIM(void *Wrapper) {
  auto *LambdaWrapper = reinterpret_cast<LambdaWrapper_ID_3DIM *>(Wrapper);
  cl::sycl::id<3> Id3Dim(cm_support::get_thread_idx(0),
                         cm_support::get_thread_idx(1),
                         cm_support::get_thread_idx(2));
  LambdaWrapper->Func(Id3Dim);
}

inline void InvokeLambda_ITEM_1DIM(void *Wrapper) {
  auto *LambdaWrapper = reinterpret_cast<LambdaWrapper_ITEM_1DIM *>(Wrapper);
  cl::sycl::item<1, false> Item1Dim = IDBuilder::createItem<1, false>(
      {cm_support::get_thread_count(0)}, /// Extent
      {cm_support::get_thread_idx(0)});  /// Index
  LambdaWrapper->Func(Item1Dim);
}

inline void InvokeLambda_ITEM_2DIM(void *Wrapper) {
  auto *LambdaWrapper = reinterpret_cast<LambdaWrapper_ITEM_2DIM *>(Wrapper);
  cl::sycl::item<2, false> Item2Dim = IDBuilder::createItem<2, false>(
      {cm_support::get_thread_count(0), /// Extent
       cm_support::get_thread_count(1)},
      {cm_support::get_thread_idx(0), /// Index
       cm_support::get_thread_idx(1)});
  LambdaWrapper->Func(Item2Dim);
}

inline void InvokeLambda_ITEM_3DIM(void *Wrapper) {
  auto *LambdaWrapper = reinterpret_cast<LambdaWrapper_ITEM_3DIM *>(Wrapper);
  cl::sycl::item<3, false> Item3Dim = IDBuilder::createItem<3, false>(
      {cm_support::get_thread_count(0), /// Extent
       cm_support::get_thread_count(1), cm_support::get_thread_count(2)},
      {cm_support::get_thread_idx(0), /// Index
       cm_support::get_thread_idx(1), cm_support::get_thread_idx(2)});

  LambdaWrapper->Func(Item3Dim);
}

inline void InvokeLambda_ITEM_OFFSET_1DIM(void *Wrapper) {
  auto *LambdaWrapper =
      reinterpret_cast<LambdaWrapper_ITEM_OFFSET_1DIM *>(Wrapper);

  cl::sycl::item<1, true> ItemOffset1Dim = IDBuilder::createItem<1, true>(
      {cm_support::get_thread_count(0)}, /// Extent
      {cm_support::get_thread_idx(0) +
       LambdaWrapper->GlobalOffset[0]}, /// Index
      {LambdaWrapper->GlobalOffset[0]}  /// Offset
  );
  LambdaWrapper->Func(ItemOffset1Dim);
}

inline void InvokeLambda_ITEM_OFFSET_2DIM(void *Wrapper) {
  auto *LambdaWrapper =
      reinterpret_cast<LambdaWrapper_ITEM_OFFSET_2DIM *>(Wrapper);

  cl::sycl::item<2, true> ItemOffset2Dim = IDBuilder::createItem<2, true>(
      {cm_support::get_thread_count(0), /// Extent
       cm_support::get_thread_count(1)},
      {cm_support::get_thread_idx(0) + LambdaWrapper->GlobalOffset[0], /// Index
       cm_support::get_thread_idx(1) + LambdaWrapper->GlobalOffset[1]},
      {LambdaWrapper->GlobalOffset[0], /// Offset
       LambdaWrapper->GlobalOffset[1]});
  LambdaWrapper->Func(ItemOffset2Dim);
}

inline void InvokeLambda_ITEM_OFFSET_3DIM(void *Wrapper) {
  auto *LambdaWrapper =
      reinterpret_cast<LambdaWrapper_ITEM_OFFSET_3DIM *>(Wrapper);

  cl::sycl::item<3, true> ItemOffset3Dim = IDBuilder::createItem<3, true>(
      {cm_support::get_thread_count(0), /// Extent
       cm_support::get_thread_count(1), cm_support::get_thread_count(2)},
      {cm_support::get_thread_idx(0) + LambdaWrapper->GlobalOffset[0], /// Index
       cm_support::get_thread_idx(1) + LambdaWrapper->GlobalOffset[1],
       cm_support::get_thread_idx(2) + LambdaWrapper->GlobalOffset[2]},
      {LambdaWrapper->GlobalOffset[0], /// Offset
       LambdaWrapper->GlobalOffset[1], LambdaWrapper->GlobalOffset[2]});
  LambdaWrapper->Func(ItemOffset3Dim);
}

inline void InvokeLambda_NDITEM_1DIM(void *Wrapper) {
  auto *LambdaWrapper = reinterpret_cast<LambdaWrapper_NDITEM_1DIM *>(Wrapper);
  sycl::range<1> GroupSize(
      sycl::detail::InitializedVal<1, sycl::range>::template get<0>());

  if (LambdaWrapper->LocalSize[0] == 0 ||
      LambdaWrapper->GlobalSize[0] % LambdaWrapper->LocalSize[0] != 0) {
    throw sycl::nd_range_error("Invalid local size for global size - 1DIM",
                               PI_INVALID_WORK_GROUP_SIZE);
  }
  GroupSize[0] = LambdaWrapper->GlobalSize[0] / LambdaWrapper->LocalSize[0];

  const sycl::id<1> LocalID = {cm_support::get_thread_idx(0)};

  const sycl::id<1> GroupID = {cm_support::get_group_idx(0)};

  const sycl::group<1> Group = IDBuilder::createGroup<1>(
      LambdaWrapper->GlobalSize, LambdaWrapper->LocalSize, GroupSize, GroupID);

  const sycl::id<1> GlobalID = GroupID * LambdaWrapper->LocalSize + LocalID +
                               LambdaWrapper->GlobalOffset;
  const sycl::item<1, /*Offset=*/true> GlobalItem =
      IDBuilder::createItem<1, true>(LambdaWrapper->GlobalSize, GlobalID,
                                     LambdaWrapper->GlobalOffset);
  const sycl::item<1, /*Offset=*/false> LocalItem =
      IDBuilder::createItem<1, false>(LambdaWrapper->LocalSize, LocalID);

  const sycl::nd_item<1> NDItem1Dim =
      IDBuilder::createNDItem<1>(GlobalItem, LocalItem, Group);

  LambdaWrapper->Func(NDItem1Dim);
}

inline void InvokeLambda_NDITEM_2DIM(void *Wrapper) {
  auto *LambdaWrapper = reinterpret_cast<LambdaWrapper_NDITEM_2DIM *>(Wrapper);
  sycl::range<2> GroupSize(
      sycl::detail::InitializedVal<2, sycl::range>::template get<0>());

  for (int I = 0; I < 2 /*Dims*/; ++I) {
    if (LambdaWrapper->LocalSize[I] == 0 ||
        LambdaWrapper->GlobalSize[I] % LambdaWrapper->LocalSize[I] != 0) {
      throw sycl::nd_range_error("Invalid local size for global size - 2DIM",
                                 PI_INVALID_WORK_GROUP_SIZE);
    }
    GroupSize[I] = LambdaWrapper->GlobalSize[I] / LambdaWrapper->LocalSize[I];
  }

  const sycl::id<2> LocalID = {cm_support::get_thread_idx(0),
                               cm_support::get_thread_idx(1)};

  const sycl::id<2> GroupID = {cm_support::get_group_idx(0),
                               cm_support::get_group_idx(1)};

  const sycl::group<2> Group = IDBuilder::createGroup<2>(
      LambdaWrapper->GlobalSize, LambdaWrapper->LocalSize, GroupSize, GroupID);

  const sycl::id<2> GlobalID = GroupID * LambdaWrapper->LocalSize + LocalID +
                               LambdaWrapper->GlobalOffset;
  const sycl::item<2, /*Offset=*/true> GlobalItem =
      IDBuilder::createItem<2, true>(LambdaWrapper->GlobalSize, GlobalID,
                                     LambdaWrapper->GlobalOffset);
  const sycl::item<2, /*Offset=*/false> LocalItem =
      IDBuilder::createItem<2, false>(LambdaWrapper->LocalSize, LocalID);

  const sycl::nd_item<2> NDItem2Dim =
      IDBuilder::createNDItem<2>(GlobalItem, LocalItem, Group);

  LambdaWrapper->Func(NDItem2Dim);
}

inline void InvokeLambda_NDITEM_3DIM(void *Wrapper) {
  auto *LambdaWrapper = reinterpret_cast<LambdaWrapper_NDITEM_3DIM *>(Wrapper);
  sycl::range<3> GroupSize(
      sycl::detail::InitializedVal<3, sycl::range>::template get<0>());

  for (int I = 0; I < 3 /*Dims*/; ++I) {
    if (LambdaWrapper->LocalSize[I] == 0 ||
        LambdaWrapper->GlobalSize[I] % LambdaWrapper->LocalSize[I] != 0) {
      throw sycl::nd_range_error("Invalid local size for global size - 3DIM",
                                 PI_INVALID_WORK_GROUP_SIZE);
    }
    GroupSize[I] = LambdaWrapper->GlobalSize[I] / LambdaWrapper->LocalSize[I];
  }

  const sycl::id<3> LocalID = {cm_support::get_thread_idx(0),
                               cm_support::get_thread_idx(1),
                               cm_support::get_thread_idx(2)};

  const sycl::id<3> GroupID = {cm_support::get_group_idx(0),
                               cm_support::get_group_idx(1),
                               cm_support::get_group_idx(2)};

  const sycl::group<3> Group = IDBuilder::createGroup<3>(
      LambdaWrapper->GlobalSize, LambdaWrapper->LocalSize, GroupSize, GroupID);

  const sycl::id<3> GlobalID = GroupID * LambdaWrapper->LocalSize + LocalID +
                               LambdaWrapper->GlobalOffset;
  const sycl::item<3, /*Offset=*/true> GlobalItem =
      IDBuilder::createItem<3, true>(LambdaWrapper->GlobalSize, GlobalID,
                                     LambdaWrapper->GlobalOffset);
  const sycl::item<3, /*Offset=*/false> LocalItem =
      IDBuilder::createItem<3, false>(LambdaWrapper->LocalSize, LocalID);

  const sycl::nd_item<3> NDItem3Dim =
      IDBuilder::createNDItem<3>(GlobalItem, LocalItem, Group);

  LambdaWrapper->Func(NDItem3Dim);
}
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

    ESimdCPUKernel ESimdCPU((fptrVoid)InvokeLambda_ID_1DIM, GroupDim, SpaceDim);

    ESimdCPU.launchMT(sizeof(struct LambdaWrapper_ID_1DIM),
                      WrappedLambda_ID_1DIM.get());
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

    ESimdCPUKernel ESimdCPU((fptrVoid)InvokeLambda_ID_2DIM, GroupDim, SpaceDim);

    ESimdCPU.launchMT(sizeof(struct LambdaWrapper_ID_2DIM),
                      WrappedLambda_ID_2DIM.get());
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

    ESimdCPUKernel ESimdCPU((fptrVoid)InvokeLambda_ID_3DIM, GroupDim, SpaceDim);

    ESimdCPU.launchMT(sizeof(struct LambdaWrapper_ID_3DIM),
                      WrappedLambda_ID_3DIM.get());
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

    ESimdCPUKernel ESimdCPU((fptrVoid)InvokeLambda_ITEM_1DIM, GroupDim,
                            SpaceDim);

    ESimdCPU.launchMT(sizeof(struct LambdaWrapper_ITEM_1DIM),
                      WrappedLambda_ITEM_1DIM.get());
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

    ESimdCPUKernel ESimdCPU((fptrVoid)InvokeLambda_ITEM_2DIM, GroupDim,
                            SpaceDim);

    ESimdCPU.launchMT(sizeof(struct LambdaWrapper_ITEM_2DIM),
                      WrappedLambda_ITEM_2DIM.get());
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

    ESimdCPUKernel ESimdCPU((fptrVoid)InvokeLambda_ITEM_3DIM, GroupDim,
                            SpaceDim);

    ESimdCPU.launchMT(sizeof(struct LambdaWrapper_ITEM_3DIM),
                      WrappedLambda_ITEM_3DIM.get());
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

    ESimdCPUKernel ESimdCPU((fptrVoid)InvokeLambda_ITEM_OFFSET_1DIM, GroupDim,
                            SpaceDim);

    ESimdCPU.launchMT(sizeof(struct LambdaWrapper_ITEM_OFFSET_1DIM),
                      WrappedLambda_ITEM_OFFSET_1DIM.get());
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

    ESimdCPUKernel ESimdCPU((fptrVoid)InvokeLambda_ITEM_OFFSET_2DIM, GroupDim,
                            SpaceDim);

    ESimdCPU.launchMT(sizeof(struct LambdaWrapper_ITEM_OFFSET_2DIM),
                      WrappedLambda_ITEM_OFFSET_2DIM.get());
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

    ESimdCPUKernel ESimdCPU((fptrVoid)InvokeLambda_ITEM_OFFSET_3DIM, GroupDim,
                            SpaceDim);

    ESimdCPU.launchMT(sizeof(struct LambdaWrapper_ITEM_OFFSET_3DIM),
                      WrappedLambda_ITEM_OFFSET_3DIM.get());
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

    ESimdCPUKernel ESimdCPU((fptrVoid)InvokeLambda_NDITEM_1DIM, GroupDim,
                            SpaceDim);

    ESimdCPU.launchMT(sizeof(struct LambdaWrapper_NDITEM_1DIM),
                      WrappedLambda_NDITEM_1DIM.get());
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

    ESimdCPUKernel ESimdCPU((fptrVoid)InvokeLambda_NDITEM_2DIM, GroupDim,
                            SpaceDim);

    ESimdCPU.launchMT(sizeof(struct LambdaWrapper_NDITEM_2DIM),
                      WrappedLambda_NDITEM_2DIM.get());
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

    ESimdCPUKernel ESimdCPU((fptrVoid)InvokeLambda_NDITEM_3DIM, GroupDim,
                            SpaceDim);

    ESimdCPU.launchMT(sizeof(struct LambdaWrapper_NDITEM_3DIM),
                      WrappedLambda_NDITEM_3DIM.get());
  }
};

/// Implementation for ESIMD_CPU device interface accessing ESIMD
/// intrinsics and LibCM functionalties requred by intrinsics
// Intrinsics
sycl::detail::ESIMDDeviceInterface::ESIMDDeviceInterface() {
  version = ESIMDEmuPluginInterfaceVersion;
  reserved = nullptr;

  /* From 'esimd_emu_functions_v1.h' : Start */
  cm_barrier_ptr = cm_support::barrier;
  cm_sbarrier_ptr = cm_support::split_barrier;
  cm_fence_ptr = cm_support::fence;

  sycl_get_surface_base_addr_ptr = cm_support::get_surface_base_addr;
  __cm_emu_get_slm_ptr = cm_support::get_slm_base;
  cm_slm_init_ptr = cm_support::init_slm;
  /* From 'esimd_emu_functions_v1.h' : End */
}

/// Implementation for Host Kernel Launch used by
/// piEnqueueKernelLaunch
template <typename T> using KernelFunc = std::function<void(T)>;

template <int NDims> struct InvokeBaseImpl {
  static sycl::range<NDims> get_range(const size_t *GlobalWorkSize);
};

static constexpr bool isNull(int NDims, const size_t *R) {
  return ((0 == R[0]) && (1 > NDims || 0 == R[1]) && (2 > NDims || 0 == R[2]));
}

template <int NDims, typename ArgTy> struct InvokeImpl {

  template <int _NDims = NDims>
  static typename std::enable_if<_NDims == 1, sycl::range<1>>::type
  get_range(const size_t *a) {
    return sycl::range<1>(a[0]);
  }

  template <int _NDims = NDims>
  static typename std::enable_if<_NDims == 2, sycl::range<2>>::type
  get_range(const size_t *a) {
    return sycl::range<2>(a[0], a[1]);
  }

  template <int _NDims = NDims>
  static typename std::enable_if<_NDims == 3, sycl::range<3>>::type
  get_range(const size_t *a) {
    return sycl::range<3>(a[0], a[1], a[2]);
  }

  static void invoke(void *fptr, const sycl::range<NDims> &range) {
    auto f = reinterpret_cast<std::function<void(const ArgTy &)> *>(fptr);
    libCMBatch<KernelFunc<const ArgTy &>, ArgTy, NDims> CmThreading(*f);
    CmThreading.runIterationSpace(range);
  }

  static void invoke(void *fptr, const size_t *GlobalWorkSize) {
    sycl::range<NDims> range = get_range(GlobalWorkSize);
    invoke(fptr, range);
  }

  static void invoke(void *fptr, const size_t *GlobalWorkOffset,
                     const size_t *GlobalWorkSize) {
    auto GlobalSize = get_range(GlobalWorkSize);
    sycl::id<NDims> GlobalOffset = get_range(GlobalWorkOffset);

    auto f = reinterpret_cast<std::function<void(const ArgTy &)> *>(fptr);
    libCMBatch<KernelFunc<const ArgTy &>, ArgTy, NDims> CmThreading(*f);
    CmThreading.runIterationSpace(GlobalSize, GlobalOffset);
  }

  static void invoke(void *fptr, const size_t *GlobalWorkOffset,
                     const size_t *GlobalWorkSize,
                     const size_t *LocalWorkSize) {
    const size_t LocalWorkSz[] = {1, 1, 1};
    if (isNull(NDims, LocalWorkSize)) {
      LocalWorkSize = LocalWorkSz;
    }

    auto GlobalSize = get_range(GlobalWorkSize);
    auto LocalSize = get_range(LocalWorkSize);
    sycl::id<NDims> GlobalOffset = get_range(GlobalWorkOffset);

    auto f = reinterpret_cast<std::function<void(const ArgTy &)> *>(fptr);
    libCMBatch<KernelFunc<const ArgTy &>, ArgTy, NDims> CmThreading(*f);

    CmThreading.runIterationSpace(LocalSize, GlobalSize, GlobalOffset);
  }
};

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
  if (NumEntries == 0 && Platforms != nullptr) {
    return PI_INVALID_VALUE;
  }
  if (Platforms == nullptr && NumPlatforms == nullptr) {
    return PI_INVALID_VALUE;
  }

  if (Platforms && NumEntries > 0) {
    *Platforms = new _pi_platform();
    Platforms[0]->CmEmuVersion = std::string("0.0.1");
  }

  if (NumPlatforms) {
    *NumPlatforms = 1;
  }

  return PI_SUCCESS;
}

pi_result piPlatformGetInfo(pi_platform Platform, pi_platform_info ParamName,
                            size_t ParamValueSize, void *ParamValue,
                            size_t *ParamValueSizeRet) {
  assert(Platform);
  ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);

  switch (ParamName) {
  case PI_PLATFORM_INFO_NAME:
    return ReturnValue("Intel(R) ESIMD_CPU/GPU");

  case PI_PLATFORM_INFO_VENDOR:
    return ReturnValue("Intel(R) Corporation");

  case PI_PLATFORM_INFO_VERSION:
    return ReturnValue(Platform->CmEmuVersion);

  case PI_PLATFORM_INFO_PROFILE:
    return ReturnValue("CM_FULL_PROFILE");

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
  return PI_SUCCESS;
}

pi_result piextPlatformCreateWithNativeHandle(pi_native_handle, pi_platform *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piDevicesGet(pi_platform Platform, pi_device_type DeviceType,
                       pi_uint32 NumEntries, pi_device *Devices,
                       pi_uint32 *NumDevices) {
  if (NumEntries == 0) {
    if (NumDevices) {
      *NumDevices = 1;
    } else {
      return PI_INVALID_VALUE;
    }
  }

  if (NumDevices) {
    *NumDevices = 1;
  } else {
    // assert(NumEntries == 1);
    Devices[0] = new _pi_device(Platform);
  }

  return PI_SUCCESS;
}

pi_result piDeviceRetain(pi_device Device) {
  assert(Device);

  ++(Device->RefCount);

  return PI_SUCCESS;
}

pi_result piDeviceRelease(pi_device) {
  CONTINUE_NO_IMPLEMENTATION;
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
    return ReturnValue("ESIMD_CPU");
  case PI_DEVICE_INFO_IMAGE_SUPPORT:
    return ReturnValue(pi_bool{true});
  case PI_DEVICE_INFO_DRIVER_VERSION:
    return ReturnValue("0.0.1");
  case PI_DEVICE_INFO_VENDOR:
    return ReturnValue("Intel(R) Corporation");
  case PI_DEVICE_INFO_IMAGE2D_MAX_WIDTH:
    return ReturnValue(size_t{8192});
  case PI_DEVICE_INFO_IMAGE2D_MAX_HEIGHT:
    return ReturnValue(size_t{8192});
  case PI_DEVICE_INFO_HOST_UNIFIED_MEMORY:
    return ReturnValue(pi_bool{1});

#define UNSUPPORTED_INFO(info)                                                 \
  case info:                                                                   \
    std::cerr << std::endl                                                     \
              << "Unsupported defice info = " << #info << std::endl;           \
    DIE_NO_IMPLEMENTATION;                                                     \
    break;

    UNSUPPORTED_INFO(PI_DEVICE_INFO_VENDOR_ID)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_EXTENSIONS)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_COMPILER_AVAILABLE)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_LINKER_AVAILABLE)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_MAX_COMPUTE_UNITS)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_MAX_WORK_GROUP_SIZE)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_MAX_WORK_ITEM_SIZES)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_MAX_CLOCK_FREQUENCY)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_ADDRESS_BITS)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_MAX_MEM_ALLOC_SIZE)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_GLOBAL_MEM_SIZE)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_LOCAL_MEM_SIZE)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_AVAILABLE)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_VERSION)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_PARTITION_MAX_SUB_DEVICES)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_REFERENCE_COUNT)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_PARTITION_PROPERTIES)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_PARTITION_TYPE)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_OPENCL_C_VERSION)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_PREFERRED_INTEROP_USER_SYNC)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_PRINTF_BUFFER_SIZE)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_PROFILE)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_BUILT_IN_KERNELS)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_QUEUE_PROPERTIES)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_EXECUTION_CAPABILITIES)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_ENDIAN_LITTLE)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_ERROR_CORRECTION_SUPPORT)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_PROFILING_TIMER_RESOLUTION)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_LOCAL_MEM_TYPE)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_MAX_CONSTANT_ARGS)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_MAX_CONSTANT_BUFFER_SIZE)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_GLOBAL_MEM_CACHE_TYPE)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_GLOBAL_MEM_CACHELINE_SIZE)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_MAX_PARAMETER_SIZE)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_MEM_BASE_ADDR_ALIGN)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_MAX_SAMPLERS)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_MAX_READ_IMAGE_ARGS)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_MAX_WRITE_IMAGE_ARGS)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_SINGLE_FP_CONFIG)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_HALF_FP_CONFIG)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_DOUBLE_FP_CONFIG)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_IMAGE3D_MAX_WIDTH)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_IMAGE3D_MAX_HEIGHT)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_IMAGE3D_MAX_DEPTH)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_IMAGE_MAX_BUFFER_SIZE)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_CHAR)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_SHORT)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_INT)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_LONG)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_FLOAT)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_DOUBLE)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_HALF)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_MAX_NUM_SUB_GROUPS)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_SUB_GROUP_SIZES_INTEL)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_IL_VERSION)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_USM_HOST_SUPPORT)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_USM_DEVICE_SUPPORT)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT)

#undef UNSUPPORTED_INFO
  default:
    DIE_NO_IMPLEMENTATION;
  }
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
  if (NumDevices != 1) {
    return PI_INVALID_VALUE;
  }
  assert(Devices);
  assert(RetContext);

  cm_support::CmDevice *device = nullptr;
  unsigned int version = 0;

  int result = cm_support::CreateCmDevice(device, version);

  if (result != cm_support::CM_SUCCESS) {
    return PI_INVALID_VALUE;
  }

  try {
    *RetContext = new _pi_context(*Devices, device);
  } catch (const std::bad_alloc &) {
    return PI_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }
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
  if ((Context == nullptr) || (Context->CmDevicePtr == nullptr)) {
    return PI_INVALID_CONTEXT;
  }

  int result = cm_support::DestroyCmDevice(Context->CmDevicePtr);
  if (result != cm_support::CM_SUCCESS) {
    return PI_INVALID_CONTEXT;
  }

  delete Context;
  return PI_SUCCESS;
}

pi_result piQueueCreate(pi_context Context, pi_device Device,
                        pi_queue_properties Properties, pi_queue *Queue) {
  cm_support::CmQueue *cmQueue;

  int result = Context->CmDevicePtr->CreateQueue(cmQueue);
  if (result != cm_support::CM_SUCCESS) {
    return PI_INVALID_CONTEXT;
  }

  try {
    *Queue = new _pi_queue(Context, cmQueue);
  } catch (const std::bad_alloc &) {
    return PI_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }

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
  if ((Queue == nullptr) || (Queue->CmQueuePtr == nullptr)) {
    return PI_INVALID_QUEUE;
  }

  // TODO : Destory 'Queue->CmQueuePtr'?
  delete Queue;

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
                                           pi_queue *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piMemBufferCreate(pi_context Context, pi_mem_flags Flags, size_t Size,
                            void *HostPtr, pi_mem *RetMem,
                            const pi_mem_properties *properties) {
  assert((Flags & PI_MEM_FLAGS_ACCESS_RW) != 0);
  assert(Context);
  assert(RetMem);

  cm_support::CmBuffer *CmBuf = nullptr;
  cm_support::SurfaceIndex *CmIndex;

  int status = Context->CmDevicePtr->CreateBuffer(
      static_cast<unsigned int>(Size), CmBuf);

  if (status != cm_support::CM_SUCCESS) {
    return PI_OUT_OF_HOST_MEMORY;
  }

  status = CmBuf->GetIndex(CmIndex);

  if ((Flags & PI_MEM_FLAGS_HOST_PTR_USE) != 0 ||
      (Flags & PI_MEM_FLAGS_HOST_PTR_COPY) != 0) {
    status =
        CmBuf->WriteSurface(reinterpret_cast<const unsigned char *>(HostPtr),
                            nullptr, static_cast<unsigned int>(Size));
  }

  auto HostPtrOrNull =
      (Flags & PI_MEM_FLAGS_HOST_PTR_USE) ? pi_cast<char *>(HostPtr) : nullptr;

  try {
    *RetMem =
        new _pi_buffer(Context, HostPtrOrNull, CmBuf,
                       /* integer buffer index */ CmIndex->get_data(), Size);
  } catch (const std::bad_alloc &) {
    return PI_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }

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
  if (Mem->getMemType() == PI_MEM_TYPE_BUFFER) {
    _pi_buffer *pi_buf = static_cast<_pi_buffer *>(Mem);
    int result = Mem->Context->CmDevicePtr->DestroySurface(pi_buf->CmBufferPtr);

    if (result != cm_support::CM_SUCCESS) {
      return PI_INVALID_MEM_OBJECT;
    }
  } else if (Mem->getMemType() == PI_MEM_TYPE_IMAGE2D) {
    _pi_image *pi_image = static_cast<_pi_image *>(Mem);
    int result =
        Mem->Context->CmDevicePtr->DestroySurface(pi_image->CmSurfacePtr);

    if (result != cm_support::CM_SUCCESS) {
      return PI_INVALID_MEM_OBJECT;
    }
  } else if (Mem->getMemType() == PI_MEM_TYPE_IMAGE2D) {
    _pi_image *pi_img = static_cast<_pi_image *>(Mem);
    int result =
        Mem->Context->CmDevicePtr->DestroySurface(pi_img->CmSurfacePtr);
    if (result != cm_support::CM_SUCCESS) {
      return PI_INVALID_MEM_OBJECT;
    }
  } else {
    return PI_INVALID_MEM_OBJECT;
  }

  return PI_SUCCESS;
}

cm_support::CM_SURFACE_FORMAT
piImageFormatToCmFormat(const pi_image_format *piFormat) {
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
  auto result = pi2cm.find(
      {piFormat->image_channel_data_type, piFormat->image_channel_order});
  if (result != pi2cm.end()) {
    return result->second;
  }
  DIE_NO_IMPLEMENTATION;
  return cm_support::CM_SURFACE_FORMAT_A8R8G8B8;
}

pi_result piMemImageCreate(pi_context Context, pi_mem_flags Flags,
                           const pi_image_format *ImageFormat,
                           const pi_image_desc *ImageDesc, void *HostPtr,
                           pi_mem *RetImage) {
  if (ImageFormat == nullptr || ImageDesc == nullptr)
    return PI_INVALID_VALUE;

  switch (ImageDesc->image_type) {
  case PI_MEM_TYPE_IMAGE2D:
    break;
  case PI_MEM_TYPE_IMAGE1D_BUFFER:
    // NOTE : Temporarily added for enabling vadd_1d and
    // vadd_raw_send. Remove for migration to github as
    // 'wrapIntoImageBuffer' is deprecated in github repo
    assert(ImageFormat->image_channel_data_type ==
           PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8);
    assert(ImageFormat->image_channel_order == PI_IMAGE_CHANNEL_ORDER_R);
    assert(ImageDesc->image_height == 0);
    return piMemBufferCreate(Context, Flags, ImageDesc->image_width, HostPtr,
                             RetImage);
  default:
    return PI_INVALID_MEM_OBJECT;
  }

  auto bytesPerPixel = 4;
  switch (ImageFormat->image_channel_data_type) {
  case PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32:
    bytesPerPixel = 16;
    break;
  case PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8:
  case PI_IMAGE_CHANNEL_TYPE_UNORM_INT8:
    bytesPerPixel = 4;
    break;
  default:
    return PI_INVALID_VALUE;
  }

  cm_support::CmSurface2D *CmSurface = nullptr;
  cm_support::SurfaceIndex *CmIndex;

  int status = Context->CmDevicePtr->CreateSurface2D(
      static_cast<unsigned int>(ImageDesc->image_width),
      static_cast<unsigned int>(ImageDesc->image_height),
      piImageFormatToCmFormat(ImageFormat), CmSurface);

  if (status != cm_support::CM_SUCCESS) {
    return PI_OUT_OF_HOST_MEMORY;
  }

  status = CmSurface->GetIndex(CmIndex);

  if ((Flags & PI_MEM_FLAGS_HOST_PTR_USE) != 0 ||
      (Flags & PI_MEM_FLAGS_HOST_PTR_COPY) != 0) {

    if (HostPtr != nullptr) {

      status = CmSurface->WriteSurface(
          reinterpret_cast<const unsigned char *>(HostPtr), nullptr,
          static_cast<unsigned int>(ImageDesc->image_width *
                                    ImageDesc->image_height * bytesPerPixel));
    }
  }

  auto HostPtrOrNull =
      (Flags & PI_MEM_FLAGS_HOST_PTR_USE) ? pi_cast<char *>(HostPtr) : nullptr;

  try {
    *RetImage = new _pi_image(Context, HostPtrOrNull, CmSurface,
                              /* integer surface index */ CmIndex->get_data(),
                              ImageDesc->image_width, ImageDesc->image_height,
                              bytesPerPixel);
  } catch (const std::bad_alloc &) {
    return PI_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }

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
  std::cerr << "Warning : Profiling Not supported under PI_ESIMD_CPU"
            << std::endl;
  return PI_SUCCESS;
}

pi_result piEventsWait(pi_uint32 NumEvents, const pi_event *EventList) {
  for (int i = 0; i < (int)NumEvents; i++) {
    if (EventList[i]->IsDummyEvent) {
      continue;
    }
    int result = EventList[i]->CmEventPtr->WaitForTaskFinished();
    if (result != cm_support::CM_SUCCESS) {
      return PI_OUT_OF_RESOURCES;
    }
  }
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
  if (!Event->IsDummyEvent) {
    if ((Event->CmEventPtr == nullptr) || (Event->OwnerQueue == nullptr)) {
      return PI_INVALID_EVENT;
    }
    int result = Event->OwnerQueue->DestroyEvent(Event->CmEventPtr);
    if (result != cm_support::CM_SUCCESS) {
      return PI_INVALID_EVENT;
    }
  }
  delete Event;

  return PI_SUCCESS;
}

pi_result piextEventGetNativeHandle(pi_event, pi_native_handle *) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextEventCreateWithNativeHandle(pi_native_handle, pi_event *) {
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
  /// TODO : Support Blocked read, 'Queue' handling
  assert(BlockingRead == false);
  assert(NumEventsInWaitList == 0);

  _pi_buffer *buf = static_cast<_pi_buffer *>(Src);

  int status =
      buf->CmBufferPtr->ReadSurface(reinterpret_cast<unsigned char *>(Dst),
                                    nullptr, // event
                                    static_cast<uint64_t>(Size));

  if (status != cm_support::CM_SUCCESS) {
    return PI_INVALID_MEM_OBJECT;
  }

  if (Event) {
    try {
      *Event = new _pi_event();
    } catch (const std::bad_alloc &) {
      return PI_OUT_OF_HOST_MEMORY;
    } catch (...) {
      return PI_ERROR_UNKNOWN;
    }
    (*Event)->IsDummyEvent = true;
  }

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
  _pi_image *img = static_cast<_pi_image *>(image);
  int status =
      img->CmSurfacePtr->ReadSurface(reinterpret_cast<unsigned char *>(ptr),
                                     nullptr, // event
                                     row_pitch * (region->height));
  if (status != cm_support::CM_SUCCESS) {
    return PI_INVALID_MEM_OBJECT;
  }

  if (event) {
    try {
      *event = new _pi_event();
    } catch (const std::bad_alloc &) {
      return PI_OUT_OF_HOST_MEMORY;
    } catch (...) {
      return PI_ERROR_UNKNOWN;
    }
    (*event)->IsDummyEvent = true;
  }
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
  switch (WorkDim) {
  case 1:
    InvokeImpl<1, sycl::nd_item<1>>::invoke(Kernel, GlobalWorkOffset,
                                            GlobalWorkSize, LocalWorkSize);
    return PI_SUCCESS;

  case 2:
    InvokeImpl<2, sycl::nd_item<2>>::invoke(Kernel, GlobalWorkOffset,
                                            GlobalWorkSize, LocalWorkSize);
    return PI_SUCCESS;

  case 3:
    InvokeImpl<3, sycl::nd_item<3>>::invoke(Kernel, GlobalWorkOffset,
                                            GlobalWorkSize, LocalWorkSize);
    return PI_SUCCESS;

  default:
    DIE_NO_IMPLEMENTATION;
    return PI_ERROR_UNKNOWN;
  }
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
  assert(Context);
  assert(ResultPtr);

  cm_support::CmBufferSVM *buf = nullptr;
  void *pSystemMem = nullptr;
  int32_t ret = Context->CmDevicePtr->CreateBufferSVM(
      Size, pSystemMem, CM_SVM_ACCESS_FLAG_DEFAULT, buf);

  if (ret != cm_support::CM_SUCCESS) {
    return PI_OUT_OF_HOST_MEMORY;
  }
  *ResultPtr = pSystemMem;
  auto it = Context->Addr2CmBufferSVM.find(pSystemMem);
  assert(Context->Addr2CmBufferSVM.end() == it);
  Context->Addr2CmBufferSVM[pSystemMem] = buf;
  return PI_SUCCESS;
}

pi_result piextUSMFree(pi_context Context, void *Ptr) {
  assert(Context);
  assert(Ptr);

  cm_support::CmBufferSVM *buf = Context->Addr2CmBufferSVM[Ptr];
  assert(buf);
  auto count = Context->Addr2CmBufferSVM.erase(Ptr);
  assert(1 == count);
  int32_t ret = Context->CmDevicePtr->DestroyBufferSVM(buf);
  if (cm_support::CM_SUCCESS != ret) {
    return PI_ERROR_UNKNOWN;
  }
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
