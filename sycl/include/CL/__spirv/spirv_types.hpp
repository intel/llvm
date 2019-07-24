//===----------- spirv_types.hpp --- SPIRV types -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>

// TODO: include the header file with SPIR-V declarations from SPIRV-Headers
// project.

// Declarations of enums below is aligned with corresponding declarations in
// SPIRV-Headers repo with a few exceptions:
// - base types changed from uint to uint32_t
// - spv namespace renamed to __spv
namespace __spv {

enum class Scope : uint32_t {
  CrossDevice = 0,
  Device = 1,
  Workgroup = 2,
  Subgroup = 3,
  Invocation = 4,
};


enum class MemorySemanticsMask : uint32_t {
  None = 0x0,
  Acquire = 0x2,
  Release = 0x4,
  AcquireRelease = 0x8,
  SequentiallyConsistent = 0x10,
  UniformMemory = 0x40,
  SubgroupMemory = 0x80,
  WorkgroupMemory = 0x100,
  CrossWorkgroupMemory = 0x200,
  AtomicCounterMemory = 0x400,
  ImageMemory = 0x800,
};

enum class GroupOperation : uint32_t {
  Reduce = 0,
  InclusiveScan = 1,
  ExclusiveScan = 2
};

inline constexpr MemorySemanticsMask operator|(MemorySemanticsMask a,
                                               MemorySemanticsMask b) {
  return static_cast<MemorySemanticsMask>(static_cast<uint32_t>(a) |
                                          static_cast<uint32_t>(b));
}

} // namespace __spv

#ifdef __SYCL_DEVICE_ONLY__
// OpenCL pipe types
template<typename dataT>
using RPipeTy = __read_only __pipe const dataT;
template<typename dataT>
using WPipeTy = __write_only __pipe const dataT;

// Struct representing layout of pipe storage
struct ConstantPipeStorage {
  int32_t _PacketSize;
  int32_t _PacketAlignment;
  int32_t _Capacity;
};
#endif // __SYCL_DEVICE_ONLY__

// This class does not have definition, it is only predeclared here.
// The pointers to this class objects can be passed to or returned from
// SPIRV built-in functions.
// Only in such cases the class is recognized as SPIRV type __ocl_event_t.
#ifndef __SYCL_DEVICE_ONLY__
typedef void* __ocl_event_t;
typedef void* __ocl_sampler_t;
// Adding only the datatypes that can be currently used in SYCL,
// as per SYCL spec 1.2.1
typedef void *__ocl_image1d_ro_t;
typedef void *__ocl_image2d_ro_t;
typedef void *__ocl_image3d_ro_t;
typedef void *__ocl_image1d_wo_t;
typedef void *__ocl_image2d_wo_t;
typedef void *__ocl_image3d_wo_t;
typedef void *__ocl_image1d_array_ro_t;
typedef void *__ocl_image2d_array_ro_t;
typedef void *__ocl_image1d_array_wo_t;
typedef void *__ocl_image2d_array_wo_t;
#endif
