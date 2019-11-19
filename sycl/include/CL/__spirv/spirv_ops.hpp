//==---------- spirv_ops.hpp --- SPIRV operations -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/__spirv/spirv_types.hpp>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#ifdef __SYCL_DEVICE_ONLY__

template <typename ImageT, typename CoordT, typename ValT>
extern void __spirv_ImageWrite(ImageT, CoordT, ValT);

template <class ReTTT, typename ImageT, typename TempArgT>
extern ReTTT __spirv_ImageRead(ImageT, TempArgT);

template <typename ImageT, typename SampledType>
extern SampledType __spirv_SampledImage(ImageT, __ocl_sampler_t);

template <typename SampledType, typename TempRetT, typename TempArgT>
extern TempRetT __spirv_ImageSampleExplicitLod(SampledType, TempArgT, int,
                                               float);

template <typename dataT>
extern __ocl_event_t
__spirv_GroupAsyncCopy(__spv::Scope Execution, __attribute__((ocl_local)) dataT *Dest,
                       __attribute__((ocl_global)) dataT *Src, size_t NumElements, size_t Stride,
                       __ocl_event_t E) noexcept;

template <typename dataT>
extern __ocl_event_t
__spirv_GroupAsyncCopy(__spv::Scope Execution, __attribute__((ocl_global)) dataT *Dest,
                       __attribute__((ocl_local)) dataT *Src, size_t NumElements, size_t Stride,
                       __ocl_event_t E) noexcept;

#define OpGroupAsyncCopyGlobalToLocal __spirv_GroupAsyncCopy
#define OpGroupAsyncCopyLocalToGlobal __spirv_GroupAsyncCopy

// Atomic SPIR-V builtins
#define __SPIRV_ATOMIC_LOAD(AS, Type)                                          \
  extern Type __spirv_AtomicLoad(AS const Type *P, __spv::Scope S,             \
                                 __spv::MemorySemanticsMask O);
#define __SPIRV_ATOMIC_STORE(AS, Type)                                         \
  extern void __spirv_AtomicStore(AS Type *P, __spv::Scope S,                  \
                                  __spv::MemorySemanticsMask O, Type V);
#define __SPIRV_ATOMIC_EXCHANGE(AS, Type)                                      \
  extern Type __spirv_AtomicExchange(AS Type *P, __spv::Scope S,               \
                                     __spv::MemorySemanticsMask O, Type V);
#define __SPIRV_ATOMIC_CMP_EXCHANGE(AS, Type)                                  \
  extern Type __spirv_AtomicCompareExchange(                                   \
      AS Type *P, __spv::Scope S, __spv::MemorySemanticsMask E,                \
      __spv::MemorySemanticsMask U, Type V, Type C);
#define __SPIRV_ATOMIC_IADD(AS, Type)                                          \
  extern Type __spirv_AtomicIAdd(AS Type *P, __spv::Scope S,                   \
                                 __spv::MemorySemanticsMask O, Type V);
#define __SPIRV_ATOMIC_ISUB(AS, Type)                                          \
  extern Type __spirv_AtomicISub(AS Type *P, __spv::Scope S,                   \
                                 __spv::MemorySemanticsMask O, Type V);
#define __SPIRV_ATOMIC_SMIN(AS, Type)                                          \
  extern Type __spirv_AtomicSMin(AS Type *P, __spv::Scope S,                   \
                                 __spv::MemorySemanticsMask O, Type V);
#define __SPIRV_ATOMIC_UMIN(AS, Type)                                          \
  extern Type __spirv_AtomicUMin(AS Type *P, __spv::Scope S,                   \
                                 __spv::MemorySemanticsMask O, Type V);
#define __SPIRV_ATOMIC_SMAX(AS, Type)                                          \
  extern Type __spirv_AtomicSMax(AS Type *P, __spv::Scope S,                   \
                                 __spv::MemorySemanticsMask O, Type V);
#define __SPIRV_ATOMIC_UMAX(AS, Type)                                          \
  extern Type __spirv_AtomicUMax(AS Type *P, __spv::Scope S,                   \
                                 __spv::MemorySemanticsMask O, Type V);
#define __SPIRV_ATOMIC_AND(AS, Type)                                           \
  extern Type __spirv_AtomicAnd(AS Type *P, __spv::Scope S,                    \
                                __spv::MemorySemanticsMask O, Type V);
#define __SPIRV_ATOMIC_OR(AS, Type)                                            \
  extern Type __spirv_AtomicOr(AS Type *P, __spv::Scope S,                     \
                               __spv::MemorySemanticsMask O, Type V);
#define __SPIRV_ATOMIC_XOR(AS, Type)                                           \
  extern Type __spirv_AtomicXor(AS Type *P, __spv::Scope S,                    \
                                __spv::MemorySemanticsMask O, Type V);

#define __SPIRV_ATOMIC_FLOAT(AS, Type)                                         \
  __SPIRV_ATOMIC_LOAD(AS, Type)                                                \
  __SPIRV_ATOMIC_STORE(AS, Type)                                               \
  __SPIRV_ATOMIC_EXCHANGE(AS, Type)

#define __SPIRV_ATOMIC_BASE(AS, Type)                                          \
  __SPIRV_ATOMIC_FLOAT(AS, Type)                                               \
  __SPIRV_ATOMIC_CMP_EXCHANGE(AS, Type)                                        \
  __SPIRV_ATOMIC_IADD(AS, Type)                                                \
  __SPIRV_ATOMIC_ISUB(AS, Type)                                                \
  __SPIRV_ATOMIC_AND(AS, Type)                                                 \
  __SPIRV_ATOMIC_OR(AS, Type)                                                  \
  __SPIRV_ATOMIC_XOR(AS, Type)

#define __SPIRV_ATOMIC_SIGNED(AS, Type)                                        \
  __SPIRV_ATOMIC_BASE(AS, Type)                                                \
  __SPIRV_ATOMIC_SMIN(AS, Type)                                                \
  __SPIRV_ATOMIC_SMAX(AS, Type)

#define __SPIRV_ATOMIC_UNSIGNED(AS, Type)                                      \
  __SPIRV_ATOMIC_BASE(AS, Type)                                                \
  __SPIRV_ATOMIC_UMIN(AS, Type)                                                \
  __SPIRV_ATOMIC_UMAX(AS, Type)

// Helper atomic operations which select correct signed/unsigned version
// of atomic min/max based on the signed-ness of the type
#define __SPIRV_ATOMIC_MINMAX(AS, Op)                                          \
  template <typename T>                                                        \
  typename std::enable_if<std::is_signed<T>::value, T>::type                   \
      __spirv_Atomic##Op(AS T *Ptr, __spv::Scope Memory,                       \
                         __spv::MemorySemanticsMask Semantics, T Value) {      \
    return __spirv_AtomicS##Op(Ptr, Memory, Semantics, Value);                 \
  }                                                                            \
  template <typename T>                                                        \
  typename std::enable_if<!std::is_signed<T>::value, T>::type                  \
      __spirv_Atomic##Op(AS T *Ptr, __spv::Scope Memory,                       \
                         __spv::MemorySemanticsMask Semantics, T Value) {      \
    return __spirv_AtomicU##Op(Ptr, Memory, Semantics, Value);                 \
  }

#define __SPIRV_ATOMICS(macro, Arg) macro(__attribute__((ocl_global)), Arg) macro(__attribute__((ocl_local)), Arg)

__SPIRV_ATOMICS(__SPIRV_ATOMIC_FLOAT, float)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_SIGNED, int)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_SIGNED, long)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_SIGNED, long long)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_UNSIGNED, unsigned int)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_UNSIGNED, unsigned long)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_UNSIGNED, unsigned long long)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_MINMAX, Min)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_MINMAX, Max)

extern bool __spirv_GroupAll(__spv::Scope Execution, bool Predicate) noexcept;

extern bool __spirv_GroupAny(__spv::Scope Execution, bool Predicate) noexcept;

template <typename dataT>
extern dataT __spirv_GroupBroadcast(__spv::Scope Execution, dataT Value,
                                    uint32_t LocalId) noexcept;

template <typename dataT>
extern dataT __spirv_GroupIAdd(__spv::Scope Execution, __spv::GroupOperation Op,
                               dataT Value) noexcept;
template <typename dataT>
extern dataT __spirv_GroupFAdd(__spv::Scope Execution, __spv::GroupOperation Op,
                               dataT Value) noexcept;
template <typename dataT>
extern dataT __spirv_GroupUMin(__spv::Scope Execution, __spv::GroupOperation Op,
                               dataT Value) noexcept;
template <typename dataT>
extern dataT __spirv_GroupSMin(__spv::Scope Execution, __spv::GroupOperation Op,
                               dataT Value) noexcept;
template <typename dataT>
extern dataT __spirv_GroupFMin(__spv::Scope Execution, __spv::GroupOperation Op,
                               dataT Value) noexcept;
template <typename dataT>
extern dataT __spirv_GroupUMax(__spv::Scope Execution, __spv::GroupOperation Op,
                               dataT Value) noexcept;
template <typename dataT>
extern dataT __spirv_GroupSMax(__spv::Scope Execution, __spv::GroupOperation Op,
                               dataT Value) noexcept;
template <typename dataT>
extern dataT __spirv_GroupFMax(__spv::Scope Execution, __spv::GroupOperation Op,
                               dataT Value) noexcept;
template <typename dataT>
extern dataT __spirv_SubgroupShuffleINTEL(dataT Data,
                                          uint32_t InvocationId) noexcept;
template <typename dataT>
extern dataT __spirv_SubgroupShuffleDownINTEL(dataT Current, dataT Next,
                                              uint32_t Delta) noexcept;
template <typename dataT>
extern dataT __spirv_SubgroupShuffleUpINTEL(dataT Previous, dataT Current,
                                            uint32_t Delta) noexcept;
template <typename dataT>
extern dataT __spirv_SubgroupShuffleXorINTEL(dataT Data,
                                             uint32_t Value) noexcept;

template <typename dataT>
extern dataT
__spirv_SubgroupBlockReadINTEL(const __attribute__((ocl_global)) uint16_t *Ptr) noexcept;

template <typename dataT>
extern void __spirv_SubgroupBlockWriteINTEL(__attribute__((ocl_global)) uint16_t *Ptr,
                                            dataT Data) noexcept;

template <typename dataT>
extern dataT
__spirv_SubgroupBlockReadINTEL(const __attribute__((ocl_global)) uint32_t *Ptr) noexcept;

template <typename dataT>
extern void __spirv_SubgroupBlockWriteINTEL(__attribute__((ocl_global)) uint32_t *Ptr,
                                            dataT Data) noexcept;

template <typename dataT>
extern int32_t __spirv_ReadPipe(RPipeTy<dataT> Pipe, dataT *Data,
                                int32_t Size, int32_t Alignment) noexcept;
template <typename dataT>
extern int32_t __spirv_WritePipe(WPipeTy<dataT> Pipe, const dataT *Data,
                                 int32_t Size, int32_t Alignment) noexcept;
template <typename dataT>
extern void __spirv_ReadPipeBlockingINTEL(RPipeTy<dataT> Pipe, dataT *Data,
                                          int32_t Size,
                                          int32_t Alignment) noexcept;
template <typename dataT>
extern void __spirv_WritePipeBlockingINTEL(WPipeTy<dataT> Pipe,
                                           const dataT *Data, int32_t Size,
                                           int32_t Alignment) noexcept;
template <typename dataT>
extern RPipeTy<dataT> __spirv_CreatePipeFromPipeStorage_read(
    const ConstantPipeStorage *Storage) noexcept;
template <typename dataT>
extern WPipeTy<dataT> __spirv_CreatePipeFromPipeStorage_write(
    const ConstantPipeStorage *Storage) noexcept;

extern void __spirv_ocl_prefetch(const __attribute__((ocl_global)) char *Ptr,
                                 size_t NumBytes) noexcept;
#else // if !__SYCL_DEVICE_ONLY__

template <typename dataT>
extern __ocl_event_t
OpGroupAsyncCopyGlobalToLocal(__spv::Scope Execution, dataT *Dest, dataT *Src,
                              size_t NumElements, size_t Stride,
                              __ocl_event_t E) noexcept {
  for (size_t i = 0; i < NumElements; i++) {
    Dest[i] = Src[i * Stride];
  }
  // A real instance of the class is not needed, return dummy pointer.
  return nullptr;
}

template <typename dataT>
extern __ocl_event_t
OpGroupAsyncCopyLocalToGlobal(__spv::Scope Execution, dataT *Dest, dataT *Src,
                              size_t NumElements, size_t Stride,
                              __ocl_event_t E) noexcept {
  for (size_t i = 0; i < NumElements; i++) {
    Dest[i * Stride] = Src[i];
  }
  // A real instance of the class is not needed, return dummy pointer.
  return nullptr;
}

extern void __spirv_ocl_prefetch(const char *Ptr, size_t NumBytes) noexcept;

#endif // !__SYCL_DEVICE_ONLY__

extern void __spirv_ControlBarrier(__spv::Scope Execution, __spv::Scope Memory,
                                   uint32_t Semantics) noexcept;

extern void __spirv_MemoryBarrier(__spv::Scope Memory,
                                  uint32_t Semantics) noexcept;

extern void __spirv_GroupWaitEvents(__spv::Scope Execution, uint32_t NumEvents,
                                    __ocl_event_t *WaitEvents) noexcept;

