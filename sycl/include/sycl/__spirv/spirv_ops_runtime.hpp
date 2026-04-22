//==------ spirv_ops_runtime.hpp --- SPIRV runtime and misc operations ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/__spirv/spirv_ops_builtin_decls.hpp>

#include <type_traits>
#include <utility>

#ifdef __SYCL_DEVICE_ONLY__

extern __DPCPP_SYCL_EXTERNAL __SYCL_EXPORT void
__clc_BarrierInitialize(int64_t *state, int32_t expected_count) noexcept;

extern __DPCPP_SYCL_EXTERNAL __SYCL_EXPORT void
__clc_BarrierInvalidate(int64_t *state) noexcept;

extern __DPCPP_SYCL_EXTERNAL __SYCL_EXPORT int64_t
__clc_BarrierArrive(int64_t *state) noexcept;

extern __DPCPP_SYCL_EXTERNAL __SYCL_EXPORT int64_t
__clc_BarrierArriveAndDrop(int64_t *state) noexcept;

extern __DPCPP_SYCL_EXTERNAL __SYCL_EXPORT int64_t
__clc_BarrierArriveNoComplete(int64_t *state, int32_t count) noexcept;

extern __DPCPP_SYCL_EXTERNAL __SYCL_EXPORT int64_t
__clc_BarrierArriveAndDropNoComplete(int64_t *state, int32_t count) noexcept;

extern __DPCPP_SYCL_EXTERNAL __SYCL_EXPORT void
__clc_BarrierCopyAsyncArrive(int64_t *state) noexcept;

extern __DPCPP_SYCL_EXTERNAL __SYCL_EXPORT void
__clc_BarrierCopyAsyncArriveNoInc(int64_t *state) noexcept;

__SYCL_CONVERGENT__ extern __DPCPP_SYCL_EXTERNAL __SYCL_EXPORT void
__clc_BarrierWait(int64_t *state, int64_t arrival) noexcept;

extern __DPCPP_SYCL_EXTERNAL __SYCL_EXPORT bool
__clc_BarrierTestWait(int64_t *state, int64_t arrival) noexcept;

__SYCL_CONVERGENT__ extern __DPCPP_SYCL_EXTERNAL __SYCL_EXPORT void
__clc_BarrierArriveAndWait(int64_t *state) noexcept;

template <typename... Args>
extern __DPCPP_SYCL_EXTERNAL int
__spirv_ocl_printf(const __attribute__((opencl_constant)) char *Format,
                   Args... args);
template <typename... Args>
extern __DPCPP_SYCL_EXTERNAL int __spirv_ocl_printf(const char *Format,
                                                    Args... args);

// FIXME: __clc symbols are intended to be internal symbols to libclc/libspirv
// and should not be relied upon externally; consider them deprecated. We can't,
// however, explicitly declare __spirv_ocl versions of these builtins as that
// interferes with the implicit declarations provided by clang. This results in
// legitimate calls being seen as ambiguous and causing errors. Since these
// symbols are intended to expose native versions of bfloat16 builtins for
// NVPTX, we should probably just be exposing builtins with actual bfloat16
// types, not unsigned integer types.
#define __CLC_BF16(...)                                                        \
  extern __DPCPP_SYCL_EXTERNAL __SYCL_EXPORT __VA_ARGS__ __clc_fabs(           \
      __VA_ARGS__) noexcept;                                                   \
  extern __DPCPP_SYCL_EXTERNAL __SYCL_EXPORT __VA_ARGS__ __clc_fmin(           \
      __VA_ARGS__, __VA_ARGS__) noexcept;                                      \
  extern __DPCPP_SYCL_EXTERNAL __SYCL_EXPORT __VA_ARGS__ __clc_fmax(           \
      __VA_ARGS__, __VA_ARGS__) noexcept;                                      \
  extern __DPCPP_SYCL_EXTERNAL __SYCL_EXPORT __VA_ARGS__ __clc_fma(            \
      __VA_ARGS__, __VA_ARGS__, __VA_ARGS__) noexcept;

#define __CLC_BF16_SCAL_VEC(TYPE)                                              \
  __CLC_BF16(TYPE)                                                             \
  __CLC_BF16(__ocl_vec_t<TYPE, 2>)                                             \
  __CLC_BF16(__ocl_vec_t<TYPE, 3>)                                             \
  __CLC_BF16(__ocl_vec_t<TYPE, 4>)                                             \
  __CLC_BF16(__ocl_vec_t<TYPE, 8>)                                             \
  __CLC_BF16(__ocl_vec_t<TYPE, 16>)

__CLC_BF16_SCAL_VEC(uint16_t)
__CLC_BF16_SCAL_VEC(uint32_t)

#undef __CLC_BF16_SCAL_VEC
#undef __CLC_BF16

extern __DPCPP_SYCL_EXTERNAL int32_t __spirv_BuiltInGlobalHWThreadIDINTEL();
extern __DPCPP_SYCL_EXTERNAL int32_t __spirv_BuiltInSubDeviceIDINTEL();
extern __DPCPP_SYCL_EXTERNAL uint64_t __spirv_ReadClockKHR(int);

template <typename from, typename to>
extern __DPCPP_SYCL_EXTERNAL
    std::enable_if_t<std::is_integral_v<to> && std::is_unsigned_v<to>, to>
    __spirv_ConvertPtrToU(from val) noexcept;

template <typename T, int N>
extern __DPCPP_SYCL_EXTERNAL std::pair<__ocl_vec_t<T, N>, __ocl_vec_t<T, N>>
__spirv_IAddCarry(__ocl_vec_t<T, N> src0, __ocl_vec_t<T, N> src1);

template <typename T, int N>
extern __DPCPP_SYCL_EXTERNAL std::pair<__ocl_vec_t<T, N>, __ocl_vec_t<T, N>>
__spirv_ISubBorrow(__ocl_vec_t<T, N> src0, __ocl_vec_t<T, N> src1);

template <typename RetT, typename... ArgsT>
extern __DPCPP_SYCL_EXTERNAL __spv::__spirv_TaskSequenceINTEL *
__spirv_TaskSequenceCreateINTEL(RetT (*f)(ArgsT...), int Pipelined = -1,
                                int ClusterMode = -1,
                                unsigned int ResponseCapacity = 0,
                                unsigned int InvocationCapacity = 0) noexcept;

template <typename... ArgsT>
extern __DPCPP_SYCL_EXTERNAL void
__spirv_TaskSequenceAsyncINTEL(__spv::__spirv_TaskSequenceINTEL *TaskSequence,
                               ArgsT... Args) noexcept;

template <typename RetT>
extern __DPCPP_SYCL_EXTERNAL RetT __spirv_TaskSequenceGetINTEL(
    __spv::__spirv_TaskSequenceINTEL *TaskSequence) noexcept;

extern __DPCPP_SYCL_EXTERNAL void __spirv_TaskSequenceReleaseINTEL(
    __spv::__spirv_TaskSequenceINTEL *TaskSequence) noexcept;

#endif