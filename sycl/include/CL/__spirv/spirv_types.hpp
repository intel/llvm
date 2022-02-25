//===------------ spirv_types.hpp --- SPIRV types -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/defines_elementary.hpp>

#include <cstddef>
#include <cstdint>

// TODO: include the header file with SPIR-V declarations from SPIRV-Headers
// project.

// Declarations of enums below is aligned with corresponding declarations in
// SPIRV-Headers repo with a few exceptions:
// - base types changed from uint to uint32_t
// - spv namespace renamed to __spv
namespace __spv {

struct Scope {

  enum Flag : uint32_t {
    CrossDevice = 0,
    Device = 1,
    Workgroup = 2,
    Subgroup = 3,
    Invocation = 4,
  };

  constexpr Scope(Flag flag) : flag_value(flag) {}

  constexpr operator uint32_t() const { return flag_value; }

  Flag flag_value;
};

struct StorageClass {
  enum Flag : uint32_t {
    UniformConstant = 0,
    Input = 1,
    Uniform = 2,
    Output = 3,
    Workgroup = 4,
    CrossWorkgroup = 5,
    Private = 6,
    Function = 7,
    Generic = 8,
    PushConstant = 9,
    AtomicCounter = 10,
    Image = 11,
    StorageBuffer = 12,
    CallableDataKHR = 5328,
    CallableDataNV = 5328,
    IncomingCallableDataKHR = 5329,
    IncomingCallableDataNV = 5329,
    RayPayloadKHR = 5338,
    RayPayloadNV = 5338,
    HitAttributeKHR = 5339,
    HitAttributeNV = 5339,
    IncomingRayPayloadKHR = 5342,
    IncomingRayPayloadNV = 5342,
    ShaderRecordBufferKHR = 5343,
    ShaderRecordBufferNV = 5343,
    PhysicalStorageBuffer = 5349,
    PhysicalStorageBufferEXT = 5349,
    CodeSectionINTEL = 5605,
    CapabilityUSMStorageClassesINTEL = 5935,
    DeviceOnlyINTEL = 5936,
    HostOnlyINTEL = 5937,
    Max = 0x7fffffff,
  };
  constexpr StorageClass(Flag flag) : flag_value(flag) {}
  constexpr operator uint32_t() const { return flag_value; }
  Flag flag_value;
};

struct MemorySemanticsMask {

  enum Flag : uint32_t {
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

  constexpr MemorySemanticsMask(Flag flag) : flag_value(flag) {}

  constexpr operator uint32_t() const { return flag_value; }

  Flag flag_value;
};

enum class GroupOperation : uint32_t {
  Reduce = 0,
  InclusiveScan = 1,
  ExclusiveScan = 2
};

enum class MatrixLayout : uint32_t {
  RowMajor = 0,
  ColumnMajor = 1,
  PackedA = 2,
  PackedB = 3
};

enum class MatrixUse : uint32_t {
  MatrixA = 0,
  MatrixB = 1,
  Accumulator = 2,
  Unnecessary = 3
};

// TODO: replace the following W/A with a better solution when we have it.
// The following structure is used to represent the joint matrix type in the
// LLVM IR. The structure has a pointer to a multidimensional array member which
// makes the encoding of the matrix type information within the LLVM IR looks
// like this:
// %struct.__spirv_JointMatrixINTEL = type { [42 x [6 x [2 x [1 x float]]]]* }
// Note that an array cannot be of zero size but MatrixLayout and Scope
// parameters can; hence '+ 1' is added to the 3rd and 4th dimensions.
// In general, representing a matrix type information like this is a bit odd
// (especially for MatrixLayout and Scope parameters). But with the current
// tools we have in Clang, this is the only way to preserve and communicate this
// information to SPIRV translator.
// The long term solution would be to introduce a matrix type in Clang and use
// it instead of this member.
template <typename T, std::size_t R, std::size_t C, MatrixUse U, MatrixLayout L,
          Scope::Flag S = Scope::Flag::Subgroup>
struct __spirv_JointMatrixINTEL {
  T (*Value)[R][C][static_cast<size_t>(U) + 1][static_cast<size_t>(L) + 1][static_cast<size_t>(S) + 1];
};

} // namespace __spv

#ifdef __SYCL_DEVICE_ONLY__
// OpenCL pipe types
template <typename dataT>
using __ocl_RPipeTy = __attribute__((pipe("read_only"))) const dataT;
template <typename dataT>
using __ocl_WPipeTy = __attribute__((pipe("write_only"))) const dataT;

// OpenCL vector types
template <typename dataT, int dims>
using __ocl_vec_t = dataT __attribute__((ext_vector_type(dims)));

// Struct representing layout of pipe storage
// TODO: rename to __spirv_ConstantPipeStorage
struct ConstantPipeStorage {
  int32_t _PacketSize;
  int32_t _PacketAlignment;
  int32_t _Capacity;
};

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
// Arbitrary precision integer type
template <int Bits> using ap_int = _ExtInt(Bits);
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
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
#define __SYCL_SPV_IMAGE_TYPE(NAME) typedef void *__ocl_##NAME##_t

#define __SYCL_SPV_SAMPLED_AND_IMAGE_TYPE(NAME)                                \
  __SYCL_SPV_IMAGE_TYPE(NAME);                                                 \
  typedef void *__ocl_sampled_##NAME##_t

__SYCL_SPV_SAMPLED_AND_IMAGE_TYPE(image1d_ro);
__SYCL_SPV_SAMPLED_AND_IMAGE_TYPE(image2d_ro);
__SYCL_SPV_SAMPLED_AND_IMAGE_TYPE(image3d_ro);
__SYCL_SPV_IMAGE_TYPE(image1d_wo);
__SYCL_SPV_IMAGE_TYPE(image2d_wo);
__SYCL_SPV_IMAGE_TYPE(image3d_wo);
__SYCL_SPV_SAMPLED_AND_IMAGE_TYPE(image1d_array_ro);
__SYCL_SPV_SAMPLED_AND_IMAGE_TYPE(image2d_array_ro);
__SYCL_SPV_IMAGE_TYPE(image1d_array_wo);
__SYCL_SPV_IMAGE_TYPE(image2d_array_wo);

#undef __SYCL_SPV_IMAGE_TYPE
#undef __SYCL_SPV_SAMPLED_AND_IMAGE_TYPE
#endif
