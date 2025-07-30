//==----------- chunk.hpp --- SYCL extension for non-uniform groups --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/aspects.hpp>
#include <sycl/detail/spirv.hpp>
#include <sycl/detail/type_traits.hpp>
#include <sycl/exception.hpp>
#include <sycl/ext/oneapi/experimental/non_uniform_groups.hpp>
#include <sycl/ext/oneapi/sub_group_mask.hpp>
#include <sycl/group_barrier.hpp>
#include <sycl/id.hpp>
#include <sycl/memory_enums.hpp>
#include <sycl/range.hpp>
#include <sycl/sub_group.hpp>

#include <stddef.h>
#include <type_traits>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

template <size_t ChunkSize, typename ParentGroup> class chunk;
template <typename ParentGroup> class fragment;

template <size_t ChunkSize, typename ParentGroup>
#ifdef __SYCL_DEVICE_ONLY__
[[__sycl_detail__::__uses_aspects__(sycl::aspect::ext_oneapi_chunk)]]
#endif
inline std::enable_if_t<std::is_same_v<ParentGroup, sycl::sub_group> ||
                            sycl::detail::is_chunk_v<ParentGroup>,
                        chunk<ChunkSize, ParentGroup>>
chunked_partition(ParentGroup parent);

template <size_t ChunkSize, typename ParentGroup> class chunk {
public:
  using id_type = id<1>;
  using range_type = range<1>;
  using linear_id_type = typename ParentGroup::linear_id_type;
  static constexpr int dimensions = 1;
  static constexpr sycl::memory_scope fence_scope = ParentGroup::fence_scope;

  inline operator fragment<ParentGroup>() const {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(__NVPTX__)
    // make fragment from chunk's mask and properties
    return fragment<ParentGroup>(Mask, get_group_id(), get_group_range());
#else
    // or mask based on chunk membership for non-NVPTX devices
    uint32_t loc_id = __spirv_SubgroupLocalInvocationId();
    uint32_t chunk_start = (loc_id / ChunkSize) * ChunkSize;
    sub_group_mask::BitsType bits =
        ChunkSize == 32
            ? sub_group_mask::BitsType(~0)
            : ((sub_group_mask::BitsType(1) << ChunkSize) - 1) << chunk_start;
    sub_group_mask mask =
        sycl::detail::Builder::createSubGroupMask<ext::oneapi::sub_group_mask>(
            bits, __spirv_SubgroupSize());
    return fragment<ParentGroup>(mask, get_group_id(), get_group_range());
#endif
#else
    return fragment<ParentGroup>();
#endif
  }

  id_type get_group_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    if constexpr (sycl::detail::is_chunk_v<ParentGroup>)
      return __spirv_SubgroupLocalInvocationId() % ParentGroup::chunk_size /
             ChunkSize;
    else
      return __spirv_SubgroupLocalInvocationId() / ChunkSize;
#else
    return id_type(0);
#endif
  }

  id_type get_local_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_SubgroupLocalInvocationId() % ChunkSize;
#else
    return id_type(0);
#endif
  }

  range_type get_group_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    if constexpr (sycl::detail::is_chunk_v<ParentGroup>)
      return ParentGroup::chunk_size / ChunkSize;
    else
      return __spirv_SubgroupSize() / ChunkSize;
#else
    return range_type(0);
#endif
  }

  range_type get_local_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return ChunkSize;
#else
    return range_type(0);
#endif
  }

  linear_id_type get_group_linear_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return static_cast<linear_id_type>(get_group_id()[0]);
#else
    return linear_id_type(0);
#endif
  }

  linear_id_type get_local_linear_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return static_cast<linear_id_type>(get_local_id()[0]);
#else
    return linear_id_type(0);
#endif
  }

  linear_id_type get_group_linear_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return static_cast<linear_id_type>(get_group_range()[0]);
#else
    return linear_id_type(0);
#endif
  }

  linear_id_type get_local_linear_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return static_cast<linear_id_type>(get_local_range()[0]);
#else
    return linear_id_type(0);
#endif
  }

  bool leader() const {
#ifdef __SYCL_DEVICE_ONLY__
    return get_local_linear_id() == 0;
#else
    return linear_id_type(0);
#endif
  }

protected:
  static constexpr size_t chunk_size = ChunkSize;

#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  sub_group_mask Mask;

  chunk(ext::oneapi::sub_group_mask mask) : Mask(mask) {}

  ext::oneapi::sub_group_mask getMask() const { return Mask; }
#else
  chunk() {}

#ifdef __SYCL_DEVICE_ONLY__
  ext::oneapi::sub_group_mask getMask() const {
    ext::oneapi::sub_group_mask::BitsType MaskBits{0};
    MaskBits = ~MaskBits;
    MaskBits <<= ext::oneapi::sub_group_mask::max_bits - ChunkSize;
    MaskBits >>=
        ext::oneapi::sub_group_mask::max_bits -
        (((__spirv_SubgroupLocalInvocationId() / ChunkSize) + 1) * ChunkSize);
    return sycl::detail::Builder::createSubGroupMask<
        ext::oneapi::sub_group_mask>(MaskBits, __spirv_SubgroupMaxSize());
  }
#endif
#endif

  friend chunk<ChunkSize, ParentGroup>
  chunked_partition<ChunkSize, ParentGroup>(ParentGroup parent);

  friend sub_group_mask sycl::detail::GetMask<chunk<ChunkSize, ParentGroup>>(
      chunk<ChunkSize, ParentGroup> Group);

  template <size_t OtherChunkSize, typename OtherParentGroup>
  friend class chunk;
};

// Chunked partition implementation
template <size_t ChunkSize, typename ParentGroup>
inline std::enable_if_t<std::is_same_v<ParentGroup, sycl::sub_group> ||
                            sycl::detail::is_chunk_v<ParentGroup>,
                        chunk<ChunkSize, ParentGroup>>
chunked_partition([[maybe_unused]] ParentGroup parent) {
  static_assert((ChunkSize & (ChunkSize - size_t{1})) == size_t{0},
                "ChunkSize must be a power of 2.");
#ifdef __SYCL_DEVICE_ONLY__
  // sync all work-items in parent group before partitioning
  sycl::group_barrier(parent);

#if defined(__NVPTX__)
  uint32_t loc_id = parent.get_local_linear_id();
  uint32_t loc_size = parent.get_local_linear_range();
  uint32_t bits = ChunkSize == 32 ? 0xffffffff
                                  : ((1 << ChunkSize) - 1)
                                        << ((loc_id / ChunkSize) * ChunkSize);

  return chunk<ChunkSize, ParentGroup>(
      sycl::detail::Builder::createSubGroupMask<ext::oneapi::sub_group_mask>(
          bits, loc_size));
#else
  return chunk<ChunkSize, ParentGroup>();
#endif
#else
  return chunk<ChunkSize, ParentGroup>();
#endif
}

// Type traits
template <size_t ChunkSize, typename ParentGroup>
struct is_user_constructed_group<chunk<ChunkSize, ParentGroup>>
    : std::true_type {};

} // namespace ext::oneapi::experimental

template <size_t ChunkSize, typename ParentGroup>
struct is_group<ext::oneapi::experimental::chunk<ChunkSize, ParentGroup>>
    : std::true_type {};

} // namespace _V1
} // namespace sycl
