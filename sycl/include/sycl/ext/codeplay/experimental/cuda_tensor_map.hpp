//==----------------- tensor_map.hpp --- CUDA TMA interop wrappers ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>

#include <sycl/detail/export.hpp>

#define SYCL_EXT_CODEPLAY_CUDA_TENSOR_MAP 1

namespace sycl {
inline namespace _V1 {
class queue;
namespace ext::codeplay::experimental::cuda {
namespace detail {
/// An opaque type passed to the runtime used to describe the properties of an
/// image.

struct alignas(64) __tensor_copy_descriptor {
protected:
  unsigned char data[128];

public:
  // It'd be nice to shorten these enumeration names a little, but since many of
  // them start with numbers, that'd be an illegal name, and nobody is going to
  // prefer typing `tensor_copy_descriptor::interleave::sixteen` over
  // `tensor_copy_descriptor::interleave_16`. Additionally, naming the type
  // enumerations after the type they represent is sketchy since there are so
  // many variations of uint32 et al in the wild. Thus: in the name of
  // consistency all enumerators here duplicatively encode the type in their
  // names
  enum datatype : int {
    type_uint8,
    type_uint16,
    type_uint32,
    type_int32,
    type_uint64,
    type_int64,
    type_float16,
    type_float32,
    type_float64,
    type_bfloat16,
    type_float32_ftz,
    type_tfloat32,
    type_tfloat32_ftz,
  };
  enum interleave : int {
    interleave_none,
    interleave_16,
    interleave_32,
  };
  enum swizzle : int {
    swizzle_none,
    swizzle_32,
    swizzle_64,
    swizzle_128,
  };
  enum l2_promote : int {
    promote_none,
    promote_l2_64,
    promote_l2_128,
    promote_l2_256,
  };
  enum oob_fill : int {
    oob_fill_none,
    oob_fill_nan_request_zero_fma,
  };
};
} // namespace detail

struct __SYCL_EXPORT tiled_encode_map final
    : public detail::__tensor_copy_descriptor {
  tiled_encode_map() = delete;
  // Can't be constructed on device, only passed into kernels from the host
  tiled_encode_map(queue &q, void *addr, datatype type, uint32_t rank,
                   const uint64_t global_dims[/*rank*/],
                   const uint64_t global_strides[/*rank - 1*/],
                   const uint32_t box_dims[/*rank*/],
                   const uint32_t element_strides[/*rank*/],
                   interleave interleave, swizzle swizzle, l2_promote promote,
                   oob_fill oob_fill);
#ifdef __SYCL_DEVICE_ONLY__
  uintptr_t get_native_descriptor() const {
    return reinterpret_cast<uintptr_t>(this);
  }
#endif
};

struct __SYCL_EXPORT im2col_encode_map final
    : public detail::__tensor_copy_descriptor {
  im2col_encode_map() = delete;
  // Can't be constructed on device, only passed into kernels from the host
  im2col_encode_map(queue &q, datatype type, uint32_t rank, void *addr,
                    const uint64_t gmem_dims[/*rank*/],
                    const uint64_t gmem_strides[/*rank - 1*/],
                    const int32_t pixel_box_lower_corner[/*rank*/],
                    const int32_t pixel_box_upper_corner[/*rank*/],
                    uint32_t channels_per_pixel, uint32_t pixels_per_col,
                    const uint32_t element_strides[/*rank*/],
                    interleave interleave, swizzle swizzle, l2_promote promote,
                    oob_fill oob_fill);
#ifdef __SYCL_DEVICE_ONLY__
  uintptr_t get_native_descriptor() const {
    return reinterpret_cast<uintptr_t>(this);
  }
#endif
};

} // namespace ext::codeplay::experimental::cuda
} // namespace _V1
} // namespace sycl
