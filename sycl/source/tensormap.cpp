//==--------- tensormap.cpp - cuda tensormap entrypoints -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/queue_impl.hpp>
#include <sycl/ext/oneapi/experimental/cuda/tensormap.hpp>
#include <sycl/queue.hpp>

namespace sycl::ext::oneapi::experimental::tensor_map {

void tma_desc::encode_im_2_col(
    queue Queue, data_type TensorMapType, uint32_t TensorRank,
    void *GlobalAddress, const uint64_t *GlobalDim,
    const uint64_t *GlobalStrides, const int *PixelBoxLowerCorner,
    const int *PixelBoxUpperCorner, uint32_t ChannelsPerPixel,
    uint32_t PixelsPerColumn, const uint32_t *ElementStrides,
    interleave Interleave, swizzle Swizzle, l2_promotion L2Promotion,
    oob_fill OobFill) {
  getSyclObjImpl(Queue)->getPlugin()->call<urTensorMapEncodeIm2ColExp>(
      getSyclObjImpl(Queue.get_device())->getHandleRef(), TensorMapType,
      TensorRank, GlobalAddress, GlobalDim, PixelBoxLowerCorner,
      PixelBoxUpperCorner, ChannelsPerPixel, PixelsPerColumn, ElementStrides,
      SYCL_TO_UR_TYPE(Interleave), SYCL_TO_UR_TYPE(Swizzle),
      SYCL_TO_UR_TYPE(L2Promotion), SYCL_TO_UR_TYPE(OobFill), &Map);
}

void tma_desc::encode_tiled(queue Q, data_type TensorMapType,
                            uint32_t TensorRank, void *GlobalAddress,
                            const uint64_t *GlobalDim,
                            const uint64_t *GlobalStrides,
                            const uint32_t *BoxDim,
                            const uint32_t *ElementStrides,
                            interleave Interleave, swizzle Swizzle,
                            l2_promotion L2Promotion, oob_fill OobFill) {
  getSyclObjImpl(Queue)->getPlugin()->call<urTensorMapEncodeTiled>(
      getSyclObjImpl(Queue.get_device())->getHandleRef(), TensorMapType,
      TensorRank, GlobalAddress, GlobalDim, BoxDim, ElementStrides,
      SYCL_TO_UR_TYPE(Interleave), SYCL_TO_UR_TYPE(Swizzle),
      SYCL_TO_UR_TYPE(L2Promotion), SYCL_TO_UR_TYPE(OobFill), &Map);
}
} // namespace sycl::ext::oneapi::experimental::tensor_map
