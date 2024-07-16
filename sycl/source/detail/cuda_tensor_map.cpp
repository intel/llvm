//==----------------- tensor_map.cpp --- CUDA TMA interop wrappers ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
#include "detail/device_impl.hpp"
#include <sycl/aspects.hpp>
#include <sycl/detail/ur.hpp>
#include <sycl/ext/codeplay/experimental/cuda_tensor_map.hpp>
#include <sycl/queue.hpp>

namespace {
using tcd =
    sycl::ext::codeplay::experimental::cuda::detail::__tensor_copy_descriptor;

static inline ur_device_handle_t get_ur_device(sycl::queue &q) {
  return sycl::detail::getSyclObjImpl(q.get_device())->getHandleRef();
}

static inline sycl::detail::AdapterPtr get_adapter(sycl::queue &q) {
  return sycl::detail::getSyclObjImpl(q.get_device())->getAdapter();
}
// n.b. none of these enum converters have a default switch label so we get
// missing enumeration warnings if new enumerations are added to the underlying
// type
static inline ur_exp_tensor_map_data_type_flags_t
datatype_to_ur(tcd::datatype type) {
  switch (type) {
  case tcd::datatype::type_uint8:
    return UR_EXP_TENSOR_MAP_DATA_TYPE_FLAG_UINT8;
  case tcd::datatype::type_uint16:
    return UR_EXP_TENSOR_MAP_DATA_TYPE_FLAG_UINT16;
  case tcd::datatype::type_uint32:
    return UR_EXP_TENSOR_MAP_DATA_TYPE_FLAG_UINT32;
  case tcd::datatype::type_int32:
    return UR_EXP_TENSOR_MAP_DATA_TYPE_FLAG_INT32;
  case tcd::datatype::type_uint64:
    return UR_EXP_TENSOR_MAP_DATA_TYPE_FLAG_UINT64;
  case tcd::datatype::type_int64:
    return UR_EXP_TENSOR_MAP_DATA_TYPE_FLAG_INT64;
  case tcd::datatype::type_float16:
    return UR_EXP_TENSOR_MAP_DATA_TYPE_FLAG_FLOAT16;
  case tcd::datatype::type_float32:
    return UR_EXP_TENSOR_MAP_DATA_TYPE_FLAG_FLOAT32;
  case tcd::datatype::type_float64:
    return UR_EXP_TENSOR_MAP_DATA_TYPE_FLAG_FLOAT64;
  case tcd::datatype::type_bfloat16:
    return UR_EXP_TENSOR_MAP_DATA_TYPE_FLAG_BFLOAT16;
  case tcd::datatype::type_float32_ftz:
    return UR_EXP_TENSOR_MAP_DATA_TYPE_FLAG_FLOAT32_FTZ;
  case tcd::datatype::type_tfloat32:
    return UR_EXP_TENSOR_MAP_DATA_TYPE_FLAG_TFLOAT32;
  case tcd::datatype::type_tfloat32_ftz:
    return UR_EXP_TENSOR_MAP_DATA_TYPE_FLAG_TFLOAT32_FTZ;
  }
  throw sycl::exception(sycl::errc::invalid);
}

static inline ur_exp_tensor_map_interleave_flags_t
interleave_to_ur(tcd::interleave interleave) {
  switch (interleave) {
  case tcd::interleave::interleave_none:
    return UR_EXP_TENSOR_MAP_INTERLEAVE_FLAG_NONE;
  case tcd::interleave::interleave_16:
    return UR_EXP_TENSOR_MAP_INTERLEAVE_FLAG_16B;
  case tcd::interleave::interleave_32:
    return UR_EXP_TENSOR_MAP_INTERLEAVE_FLAG_32B;
  }
  throw sycl::exception(sycl::errc::invalid);
}

static inline ur_exp_tensor_map_swizzle_flags_t
swizzle_to_ur(tcd::swizzle swizzle) {
  switch (swizzle) {
  case tcd::swizzle::swizzle_none:
    return UR_EXP_TENSOR_MAP_SWIZZLE_FLAG_NONE;
  case tcd::swizzle::swizzle_32:
    return UR_EXP_TENSOR_MAP_SWIZZLE_FLAG_32B;
  case tcd::swizzle::swizzle_64:
    return UR_EXP_TENSOR_MAP_SWIZZLE_FLAG_64B;
  case tcd::swizzle::swizzle_128:
    return UR_EXP_TENSOR_MAP_SWIZZLE_FLAG_128B;
  }
  throw sycl::exception(sycl::errc::invalid);
}

static inline ur_exp_tensor_map_l2_promotion_flags_t
l2_promote_to_ur(tcd::l2_promote promote) {
  switch (promote) {
  case tcd::l2_promote::promote_none:
    return UR_EXP_TENSOR_MAP_L2_PROMOTION_FLAG_NONE;
  case tcd::l2_promote::promote_l2_64:
    return UR_EXP_TENSOR_MAP_L2_PROMOTION_FLAG_64B;
  case tcd::l2_promote::promote_l2_128:
    return UR_EXP_TENSOR_MAP_L2_PROMOTION_FLAG_128B;
  case tcd::l2_promote::promote_l2_256:
    return UR_EXP_TENSOR_MAP_L2_PROMOTION_FLAG_256B;
  }
  throw sycl::exception(sycl::errc::invalid);
}
static inline ur_exp_tensor_map_oob_fill_flags_t
oob_fill_to_ur(tcd::oob_fill fill) {
  switch (fill) {
  case tcd::oob_fill::oob_fill_none:
    return UR_EXP_TENSOR_MAP_OOB_FILL_FLAG_NONE;
  case tcd::oob_fill::oob_fill_nan_request_zero_fma:
    return UR_EXP_TENSOR_MAP_OOB_FILL_FLAG_REQUEST_ZERO_FMA;
  }
  throw sycl::exception(sycl::errc::invalid);
}
} // namespace

namespace sycl {
inline namespace _V1 {
namespace ext::codeplay::experimental::cuda {
tiled_encode_map::tiled_encode_map(queue &q, void *addr, datatype type,
                                   uint32_t rank,
                                   const uint64_t global_dims[/*rank*/],
                                   const uint64_t global_strides[/*rank - 1*/],
                                   const uint32_t box_dims[/*rank*/],
                                   const uint32_t element_strides[/*rank*/],
                                   interleave interleave, swizzle swizzle,
                                   l2_promote promote, oob_fill oob_fill) {
  // This static assertion looks a bit funny, due to some fun C++ "features".
  // We want to ensure that passing this struct around to kernels works as
  // expected (LLVM byval for aggregates in __GRID_CONSTANT__ memory). For that
  // to work, the tensor map data space must be the first member of the struct.
  // We can't use offsetof here because of visibility (only works with public
  // visibility (and it's not really legal for non POD types)).
  // We also can't compare pointer differences statically e.g. assert(this ==
  // data)
  // Thus the only thing I can think of to make this validation staticallly is
  // to assert that the size of the class is the size of its only member, which
  // guarantees the offset is zero.
  static_assert(sizeof *this == sizeof data,
                "the tensor data must be at offset zero for correct "
                "kernel parameter passing");

  if (!q.get_device().has(sycl::aspect::ext_codeplay_cuda_tensor_map)) {
    throw sycl::exception(
        make_error_code(errc::feature_not_supported),
        "Tensor maps are only supported on CUDA GPUs with SM >= 90");
  }

  auto ur_device_handle = get_ur_device(q);
  // XXX This pointer-to-pointer is gross, but the DDI layer generation doesn't
  // support opaque types because it needs to allocate the base type.
  auto *ur_tensor_map =
      reinterpret_cast<ur_exp_tensor_map_handle_t>(this->data);

  auto ur_type = datatype_to_ur(type);
  auto ur_swizzle = swizzle_to_ur(swizzle);
  auto ur_interleave = interleave_to_ur(interleave);
  auto ur_promote = l2_promote_to_ur(promote);
  auto ur_fill = oob_fill_to_ur(oob_fill);

  get_adapter(q)
      ->call<sycl::errc::invalid,
             sycl::detail::UrApiKind::urTensorMapEncodeTiledExp>(
          ur_device_handle, ur_type, rank, addr, global_dims, global_strides,
          box_dims, element_strides, ur_interleave, ur_swizzle, ur_promote,
          ur_fill, &ur_tensor_map);
}

im2col_encode_map::im2col_encode_map(
    queue &q, datatype type, uint32_t rank, void *addr,
    const uint64_t gmem_dims[/*rank*/],
    const uint64_t gmem_strides[/*rank - 1*/],
    const int32_t pixel_box_lower_corner[/*rank*/],
    const int32_t pixel_box_upper_corner[/*rank*/], uint32_t channels_per_pixel,
    uint32_t pixels_per_col, const uint32_t element_strides[/*rank*/],
    interleave interleave, swizzle swizzle, l2_promote promote,
    oob_fill oob_fill) {
  auto ur_device_handle = get_ur_device(q);
  // XXX This pointer-to-pointer is gross, but the DDI layer generation doesn't
  // support opaque types because it needs to allocate the base type.
  auto *ur_tensor_map =
      reinterpret_cast<ur_exp_tensor_map_handle_t>(this->data);

  auto ur_type = datatype_to_ur(type);
  auto ur_swizzle = swizzle_to_ur(swizzle);
  auto ur_interleave = interleave_to_ur(interleave);
  auto ur_promote = l2_promote_to_ur(promote);
  auto ur_fill = oob_fill_to_ur(oob_fill);
  get_adapter(q)
      ->call<sycl::errc::runtime,
             sycl::detail::UrApiKind::urTensorMapEncodeIm2ColExp>(
          ur_device_handle, ur_type, rank, addr, gmem_dims, gmem_strides,
          pixel_box_lower_corner, pixel_box_upper_corner, channels_per_pixel,
          pixels_per_col, element_strides, ur_interleave, ur_swizzle,
          ur_promote, ur_fill, &ur_tensor_map);
}
} // namespace ext::codeplay::experimental::cuda
} // namespace _V1
} // namespace sycl
