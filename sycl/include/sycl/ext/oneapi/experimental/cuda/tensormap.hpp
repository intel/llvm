//==--------- tensormap.hpp - cuda tensormap entrypoints -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/types.hpp>

namespace sycl::ext::oneapi::experimental::tensor_map {

enum class data_type {
  uint8,
  uint16,
  uint32,
  int32,
  uint64,
  int64,
  float64,
  bfloat16,
  float32_ftz,
  tfloat32,
  tfloat32_ftz,
};

enum class interleave {
  interleave_none,
  interleave_16b,
  interleave_32b,
};

enum class swizzle {
  swizzle_none,
  swizzle_32b,
  swizzle_64b,
  swizzle_128b,
};

enum class l2_promotion {
  l2_promotion_none,
  l2_promotion_64b,
  l2_promotion_128b,
  l2_promotion_256b,
};

enum class oob_fill {
  none,
  request_zero_fma,
};

class tma_desc {
  ur_tensor_map_handle_t Map; // This is equivalent to a CUtensorMap*
                              // This fails because ur.hpp isn't included in
                              // SYCL runtime yet
public:
  void encode_im_2_col(queue Q, data_type TensorMapType, uint32_t TensorRank,
                       void *GlobalAddress, const uint64_t *GlobalDim,
                       const uint64_t *GlobalStrides,
                       const int *PixelBoxLowerCorner,
                       const int *PixelBoxUpperCorner,
                       uint32_t ChannelsPerPixel, uint32_t PixelsPerColumn,
                       const uint32_t *ElementStrides, interleave Interleave,
                       swizzle Swizzle, l2_promotion L2Promotion,
                       oob_fill OobFill);

  void encode_tiled(queue Q, data_type TensorMapType, uint32_t TensorRank,
                    void *GlobalAddress, const uint64_t *GlobalDim,
                    const uint64_t *GlobalStrides, const uint32_t *BoxDim,
                    const uint32_t *ElementStrides, interleave Interleave,
                    swizzle Swizzle, l2_promotion L2Promotion,
                    oob_fill OobFill);

#ifdef __NVPTX__
  using Ret = CUTensorMap;
#else
  using Ret = struct alignas(64) { char bytes[128]; };
#endif

  // This returned const ref struct is to be used on device in some manner.
  // This may need to change.
  const Ret &get_map() { return *reinterpret_cast<Ret *>(Map); }
};

} // namespace sycl::ext::oneapi::experimental::tensor_map
