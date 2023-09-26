//==---- group_load_store.hpp --- SYCL extension for group loads/stores ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements https://github.com/intel/llvm/pull/7593

#pragma once

#include <sycl/ext/oneapi/properties/properties.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

// TODO: Should that go into other place (shared between different extensions)?
enum class group_algorithm_data_placement { blocked, striped };

namespace property {
struct data_placement_key {
  template <group_algorithm_data_placement Placement>
  using value_t =
      property_value<data_placement_key,
                     std::integral_constant<int, static_cast<int>(Placement)>>;
};

template <group_algorithm_data_placement Placement>
inline constexpr data_placement_key::value_t<Placement> data_placement;

// TODO: Include into the extension spec or remove. If latter, we'd probably
// introduce internal macro to make the same assumption globaly for the testing
// purposes.
struct full_sg_key {
  using value_t = property_value<full_sg_key>;
};

inline constexpr full_sg_key::value_t full_sg;
} // namespace property

template <>
struct is_property_key<property::data_placement_key> : std::true_type {};

template <> struct is_property_key<property::full_sg_key> : std::true_type {};

namespace detail {
using namespace sycl::detail;

template <> struct PropertyToKind<property::data_placement_key> {
  static constexpr PropKind Kind = PropKind::DataPlacement;
};
template <>
struct IsCompileTimeProperty<property::data_placement_key> : std::true_type {};

template <> struct PropertyToKind<property::full_sg_key> {
  static constexpr PropKind Kind = PropKind::FullSG;
};
template <>
struct IsCompileTimeProperty<property::full_sg_key> : std::true_type {};

// Implementation helpers.

#ifdef __SYCL_DEVICE_ONLY__
template <typename Properties> constexpr bool isBlocked(Properties properties) {
  namespace property = ext::oneapi::experimental::property;
  if constexpr (properties
                    .template has_property<property::data_placement_key>())
    return properties.template get_property<property::data_placement_key>() ==
           property::data_placement<
               ext::oneapi::experimental::group_algorithm_data_placement::
                   blocked>;
  else
    return true;
}

template <bool IsBlocked, int VEC_OR_ARRAY_SIZE, typename GroupTy>
int get_mem_idx(GroupTy g, int vec_or_array_idx) {
  if constexpr (IsBlocked)
    return g.get_local_linear_id() * VEC_OR_ARRAY_SIZE + vec_or_array_idx;
  else
    return g.get_local_linear_id() +
           g.get_local_linear_range() * vec_or_array_idx;
}

template <std::size_t type_size, std::size_t ElementsPerWorkItem, bool blocked>
inline constexpr bool no_shuffle_impl_available =
    is_power_of_two(type_size) && type_size <= 8 &&
    (!blocked || (is_power_of_two(ElementsPerWorkItem) &&
                  (ElementsPerWorkItem <= 8 ||
                   (ElementsPerWorkItem == 16 && type_size == 1))));

// A helper for group_load/store implementations outlining the common logic:
//
// - fallback to "generic" if block loads/stores aren't supported (non SPIRV
//   device, needs polishing though).
// - fallback to "generic" if not properly aligned.
// - fallback to "generic" if iterator isn't a pointer/multi_ptr.
// - fallback to "generic" if there are masked-out SIMD lanes (i.e.,
//   sg.get_local_range() isn't equal to SIMD width).
// - ensure multi_ptr is "delegated" as a plain annotated pointer.
// - use dynamic address space cast for the pointers in generic space then
//   either "delegate" or fallback to "generic".
// - finally, if the pointer is in the global address space and satisfies
//   alignment conditions, use "impl" callback to perform optimized load/store.
//
// Note that the tests for the functionality assume existence of this helper to
// avoid combinatorial explosion of scenarios to test.
template <int required_align, bool assume_full_sg, bool unsupported,
          typename GroupHelper, typename IteratorT, typename GenericTy,
          typename DelegateTy, typename ImplTy>
void dispatch_ptr(GroupHelper gh, IteratorT iter, GenericTy generic,
                  DelegateTy delegate, ImplTy impl) {
  auto group = [&]() {
    if constexpr (ext::oneapi::experimental::detail::is_group_helper_v<
                      GroupHelper>)
      return gh.get_group();
    else
      return gh;
  }();
  using GroupTy = decltype(group);

  using value_type =
      remove_decoration_t<typename std::iterator_traits<IteratorT>::value_type>;
  using iter_no_cv = std::remove_cv_t<IteratorT>;

#if defined(__SPIR__)
  constexpr bool is_spir = true;
#else
  constexpr bool is_spir = false;
#endif

  if constexpr (!is_spir || unsupported) {
    return generic();
  } else if constexpr (detail::is_multi_ptr_v<IteratorT>) {
    return delegate(iter.get_decorated());
  } else if constexpr (!std::is_pointer_v<iter_no_cv>) {
    return generic();
  } else {
    // TODO: Handle annotated_ptr?
    if constexpr (alignof(value_type) < required_align) {
      if ((reinterpret_cast<uintptr_t>(iter) % required_align) != 0) {
        return generic();
      }
    }

    if constexpr (!assume_full_sg) {
      if constexpr (detail::is_sub_group<GroupTy>::value) {
        if (group.get_local_range() != group.get_max_local_range())
          // Sub-group is not "full".
          return generic();
      } else if constexpr (detail::is_group<GroupTy>::value) {
        // TODO: Use get_child_group from
        // sycl_ext_oneapi_root_group extension once it is
        // implemented instead of this free function.
        auto ndi = sycl::ext::oneapi::experimental::this_nd_item<
            GroupTy::dimensions>();
        auto sg = ndi.get_sub_group();

        auto wg_size = group.get_local_range().size();
        auto simd_width = sg.get_max_local_range().size();

        if (wg_size % simd_width != 0) {
          return generic();
        }
      }
    }

    // TODO: verify that sizeof * sg.get_max_local_range() *
    // sg_offset_per_wi is a multiple of required alignment for
    // detail::is_group.

    constexpr auto AS = detail::deduce_AS<iter_no_cv>::value;
    if constexpr (AS == access::address_space::global_space) {
      // The only customization point - to be handled by the
      // caller.
      return impl(iter);
    } else if constexpr (AS == access::address_space::generic_space) {
      if (auto global_ptr =
              __SYCL_GenericCastToPtrExplicit_ToGlobal<value_type>(iter))
        return delegate(global_ptr);
      else
        return generic();
    } else {
      return generic();
    }
  }
  __builtin_unreachable();
}
#endif
} // namespace detail

#ifdef __SYCL_DEVICE_ONLY__
// Load API scalar.
template <typename Group, typename InputIteratorT, typename OutputT,
          typename Properties = decltype(properties())>
std::enable_if_t<
    std::is_convertible_v<remove_decoration_t<typename std::iterator_traits<
                              InputIteratorT>::value_type>,
                          OutputT> &&
    detail::is_generic_group_v<Group>>
group_load(Group g, InputIteratorT in_ptr, OutputT &out,
           Properties properties = {}) {
  group_load(g, in_ptr, span<OutputT, 1>(&out, 1), properties);
}

// Store API scalar.
template <typename Group, typename InputT, typename OutputIteratorT,
          typename Properties = decltype(properties())>
std::enable_if_t<std::is_convertible_v<
                     InputT, remove_decoration_t<typename std::iterator_traits<
                                 OutputIteratorT>::value_type>> &&
                 detail::is_generic_group_v<Group>>
group_store(Group g, const InputT &in, OutputIteratorT out_ptr,
            Properties properties = {}) {
  group_store(g, span<const InputT, 1>(&in, 1), out_ptr, properties);
}

// Load API sycl::vec overload.
template <typename Group, typename InputIteratorT, typename OutputT, int N,
          typename Properties = decltype(properties())>
std::enable_if_t<
    std::is_convertible_v<remove_decoration_t<typename std::iterator_traits<
                              InputIteratorT>::value_type>,
                          OutputT> &&
    detail::is_generic_group_v<Group>>
group_load(Group g, InputIteratorT in_ptr, sycl::vec<OutputT, N> &out,
           Properties properties = {}) {
  group_load(g, in_ptr, span<OutputT, N>(&out[0], N), properties);
}

// Store API sycl::vec overload.
template <typename Group, typename InputT, int N, typename OutputIteratorT,
          typename Properties = decltype(properties())>
std::enable_if_t<std::is_convertible_v<
                     InputT, remove_decoration_t<typename std::iterator_traits<
                                 OutputIteratorT>::value_type>> &&
                 detail::is_generic_group_v<Group>>
group_store(Group g, const sycl::vec<InputT, N> &in, OutputIteratorT out_ptr,
            Properties properties = {}) {
  group_store(g, span<const InputT, N>(&in[0], N), out_ptr, properties);
}

// Load API span + group/sub_group overload.
template <typename GroupHelper, typename InputIteratorT, typename OutputT,
          std::size_t ElementsPerWorkItem,
          typename Properties = decltype(properties())>
std::enable_if_t<
    std::is_convertible_v<remove_decoration_t<typename std::iterator_traits<
                              InputIteratorT>::value_type>,
                          OutputT> &&
    detail::is_generic_group_v<GroupHelper>>
group_load(GroupHelper gh, InputIteratorT in_ptr,
           span<OutputT, ElementsPerWorkItem> out, Properties properties = {}) {
  constexpr bool blocked = detail::isBlocked(properties);

  using value_type = remove_decoration_t<
      typename std::iterator_traits<InputIteratorT>::value_type>;

  // See std::enable_if_t above restricting this implementation.
  using GroupTy = GroupHelper;
  auto g = gh;

  auto generic = [&]() {
    group_barrier(g);
    detail::loop_unroll_up_to<ElementsPerWorkItem, 16>([&](size_t i) {
      auto idx = detail::get_mem_idx<blocked, ElementsPerWorkItem>(g, i);
      out[i] = in_ptr[idx];
    });
    group_barrier(g);
  };

  auto delegate = [&](auto unwrapped_ptr) {
    group_load(gh, unwrapped_ptr, out, properties);
  };

  constexpr int BlockSize =
      sizeof(value_type) * (blocked ? ElementsPerWorkItem : 1);
  constexpr int NumBlocks = blocked ? 1 : ElementsPerWorkItem;

  constexpr auto hw_block_size = [&]() {
    size_t size = 8;
    while (BlockSize % size != 0)
      size /= 2;
    return size;
  }();
  using HWBlockTy = detail::cl_unsigned<hw_block_size>;

  constexpr const size_t hw_blocks_per_block = BlockSize / hw_block_size;

  auto impl_sg = [&](sub_group sg, auto *in_ptr) {
    value_type v[ElementsPerWorkItem];

    auto priv_ptr = reinterpret_cast<char *>(&v);

    // Needs to be 4 bytes aligned (16 for writes).

    // Native is strided!
    // Available native HWBlockSizes: uchar, ushort, uint, ulong (1, 2, 4, 8).
    // Available native NumHWBlocks:
    //   1, 2, 4, 8, or 16 uchars
    //   1, 2, 4, or 8 ulongs/uints/ushorts

    size_t sg_lid = sg.get_local_linear_id();
    size_t sg_size = sg.get_max_local_range().size(); // Assume "full" SG.

    // We selected "HWBlockTy" such that sizeof(Block) % sizeof(HWBlockTy) == 0.

    //   s -> SG, w-> WI, b->hw block
    //
    // memory reads:
    //  | s0.w0.b0 | s0.w1.b0 | s0.w2.b0 | ... | sN.wS.bV |  sN.wS.bV |
    //
    // After the read, we need to rearrange data between work items and
    // write it onto each work item's own destination. For example:
    //
    //  Idx\WI |     0    |     1    |
    //    0    | s0.w0.b0 | s0.w0.b2 |
    //    1    | s0.w1.b0 | s0.w1.b2 |
    //    2    | s0.w0.b1 | s0.w0.b3 |
    //    3    | s0.w1.b1 | s0.w1.b3 |
    //         +----------+----------+
    //    0    | s0.w0.b4 | s0.w0.b6 |
    //    1    | s0.w1.b4 | s0.w1.b6 |
    //    2    | s0.w0.b5 | s0.w0.b7 |
    //    3    | s0.w1.b5 | s0.w1.b7 |

    size_t cur_hw_blocks_start_idx = 0;

    size_t cur_write_index = 0;

    // Index to the memory pointed to by the incoming argument.
    size_t needed_global_idx = sg_lid * hw_blocks_per_block;

    // select next vec_size for the load.
    // 1 == 2^0,  16 == 2^4
    constexpr size_t max_vec_pwr_of_two = hw_block_size == 1 ? 4 : 3;
    detail::loop<max_vec_pwr_of_two + 1>([&](auto i) {
      // Use bigger sizes first.
      constexpr int vec_size = 1 << (max_vec_pwr_of_two - i);

      constexpr auto iterations =
          i == 0
              ? hw_blocks_per_block * NumBlocks / vec_size
              : (hw_blocks_per_block * NumBlocks % (vec_size * 2)) / vec_size;

      detail::loop_unroll_up_to<iterations, 16>([&](auto) {
        const size_t hw_blocks_per_iter = sg_size * vec_size;

        using LoadT = std::conditional_t<
            vec_size == 1, HWBlockTy,
            detail::ConvertToOpenCLType_t<vec<HWBlockTy, vec_size>>>;

        using PtrT = typename detail::DecoratedType<
            HWBlockTy, access::address_space::global_space>::type *;
        LoadT load = __spirv_SubgroupBlockReadINTEL<LoadT>(
            reinterpret_cast<PtrT>(in_ptr) + cur_hw_blocks_start_idx);

        if constexpr (hw_blocks_per_block == 1) {
          std::memcpy(priv_ptr + cur_write_index * hw_block_size, &load,
                      sizeof(load));
          cur_write_index += vec_size;
          needed_global_idx += hw_blocks_per_block;
        } else if constexpr (detail::is_power_of_two(hw_blocks_per_block) &&
                             vec_size >= hw_blocks_per_block) {
          //  Idx\WI |     0    |     1    |
          //    0    | s0.w0.b0 | s0.w0.b2 |
          //    1    | s0.w1.b0 | s0.w1.b2 |
          //    2    | s0.w0.b1 | s0.w0.b3 |
          //    3    | s0.w1.b1 | s0.w1.b3 |

          //  Idx\WI |     0    |     1    |
          //    0    | s0.w0.b0 | s0.w0.b2 |
          //    1    | s0.w1.b0 | s0.w1.b2 |
          //    2    | s0.w0.b1 | s0.w0.b3 |
          //    3    | s0.w1.b1 | s0.w1.b3 |
          //         +----------+----------+
          //    0    | s0.w0.b4 | s0.w0.b6 |
          //    1    | s0.w1.b4 | s0.w1.b6 |
          //    2    | s0.w0.b5 | s0.w0.b7 |
          //    3    | s0.w1.b5 | s0.w1.b7 |
          detail::loop<vec_size>([&](auto i) {
            size_t BlockIdx = i / hw_blocks_per_block;
            size_t block_idx = i % hw_blocks_per_block;
            size_t idx = sg_lid * hw_blocks_per_block + block_idx +
                         BlockIdx * sg_size * hw_blocks_per_block;

            size_t wi = idx % sg_size;
            size_t block = idx / sg_size;

            auto val = select_from_group(sg, load, wi)[block];
            std::memcpy(priv_ptr + i * hw_block_size, &val, hw_block_size);
          });
        } else {
          // TODO: Verify that those shuffles are worth doing at all.
          // clang-format off
                  //  Idx\WI |     0    |     1    |     2    |     3    |     4    |     5    |     6    |     7    |
                  //    0    | s0.w0.b0 | s0.w7.b0 | s0.w6.b1 | s0.w5.b2 | s0.w4.b3 | s1.w3.b0 | s1.w2.b1 | s1.w1.b2 |
                  //    1    | s0.w1.b0 | s0.w0.b1 | s0.w7.b1 | s0.w6.b2 | s0.w5.b3 | s1.w4.b0 | s1.w3.b1 | s1.w2.b2 |
                  //    2    | s0.w2.b0 | s0.w1.b1 | s0.w0.b2 | s0.w7.b2 | s0.w6.b3 | s1.w5.b0 | s1.w4.b1 | s1.w3.b2 |
                  //    3    | s0.w3.b0 | s0.w2.b1 | s0.w1.b2 | s0.w0.b3 | s0.w7.b3 | s1.w6.b0 | s1.w5.b1 | s1.w4.b2 |
                  //    4    | s0.w4.b0 | s0.w3.b1 | s0.w2.b2 | s0.w1.b3 | s1.w0.b0 | s1.w7.b0 | s1.w6.b1 | s1.w5.b2 |
                  //    5    | s0.w5.b0 | s0.w4.b1 | s0.w3.b2 | s0.w2.b3 | s1.w1.b0 | s1.w0.b1 | s1.w7.b1 | s1.w6.b2 |
                  //    6    | s0.w6.b0 | s0.w5.b1 | s0.w4.b2 | s0.w3.b3 | s1.w2.b0 | s1.w1.b1 | s1.w0.b2 | s1.w7.b2 |
                  // ------- +----------+----------+----------+----------+----------+----------+----------+----------+
                  //    0    | s1.w0.b3 | s1.w7.b3 | s2.w6.b0 | s2.w5.b1 | s2.w4.b2 | s2.w3.b3 |
                  //    1    | s1.w1.b3 | s2.w0.b0 | s2.w7.b0 | s2.w6.b1 | s2.w5.b2 | s2.w4.b3 |
                  //    2    | s1.w2.b3 | s2.w1.b0 | s2.w0.b1 | s2.w7.b1 | s2.w6.b2 | s2.w5.b3 |
                  //    3    | s1.w3.b3 | s2.w2.b0 | s2.w1.b1 | s2.w0.b2 | s2.w7.b2 | s2.w6.b3 |
                  //    4    | s1.w4.b3 | s2.w3.b0 | s2.w2.b1 | s2.w1.b2 | s2.w0.b3 | s2.w7.b3 |
                  //    5    | s1.w5.b3 | s2.w4.b0 | s2.w3.b1 | s2.w2.b2 | s2.w1.b3 | Remainder, 16 elems
                  //    6    | s1.w6.b3 | s2.w5.b0 | s2.w4.b1 | s2.w3.b2 | s2.w2.b3 |
          // clang-format on
          while (true) {
            int needed_idx = needed_global_idx - cur_hw_blocks_start_idx;

            int wi = needed_idx % sg_size;
            int block = needed_idx / sg_size;

            // Shuffle has to be in the convergent control flow.
            auto val = select_from_group(sg, load, wi);

            bool write_needed = needed_global_idx >= cur_hw_blocks_start_idx &&
                                needed_global_idx < cur_hw_blocks_start_idx +
                                                        hw_blocks_per_iter;

            if (none_of_group(sg, write_needed))
              break;

            if (write_needed) {
              std::memcpy(priv_ptr + cur_write_index * hw_block_size,
                          reinterpret_cast<HWBlockTy *>(&val) + block,
                          hw_block_size);
              ++cur_write_index;
              needed_global_idx +=
                  cur_write_index % hw_blocks_per_block == 0
                      ? 1 + sg_size * hw_blocks_per_block - hw_blocks_per_block
                      : 1;
            }
          };
        }

        cur_hw_blocks_start_idx += hw_blocks_per_iter;
      });
    });

    // Now perform the required implicit conversion.
    detail::loop_unroll_up_to<ElementsPerWorkItem, 16>(
        [&](size_t i) { out[i] = v[i]; });
  };

  auto impl = [&](auto *in_ptr) {
    group_barrier(g);
    if constexpr (detail::is_sub_group<GroupTy>::value) {
      return impl_sg(g, in_ptr);
    } else {
      auto sg = sycl::ext::oneapi::this_sub_group();
      if constexpr (blocked) {
        return impl_sg(sg, in_ptr + sg.get_group_id() *
                                        sg.get_max_local_range() *
                                        ElementsPerWorkItem);
      } else {
        // For striped layout the stride between elements in a vector is
        // expressed in terms of WG's size, not SG. As such, each index has
        // to be implemented using scalar SG block_load.
        auto vec_elem_stride = g.get_local_linear_range();
        detail::loop_unroll_up_to<ElementsPerWorkItem, 16>([&](size_t i) {
          OutputT scalar;
          group_load(sg,
                     in_ptr + sg.get_group_id() * sg.get_max_local_range() +
                         vec_elem_stride * i,
                     scalar, properties);
          out[i] = scalar;
        });
      }
    }
    group_barrier(g);
  };

  constexpr bool assume_full_sg =
      properties.template has_property<property::full_sg_key>();
  // We'd need too much private memory.
  constexpr bool unsupported = hw_blocks_per_block > 16;
  detail::dispatch_ptr<4 /* read align in bytes */, assume_full_sg,
                       unsupported>(g, in_ptr, generic, delegate, impl);
}

// Load API span + group_helper overload.
template <typename GroupHelper, typename InputIteratorT, typename OutputT,
          std::size_t ElementsPerWorkItem,
          typename Properties = decltype(properties())>
std::enable_if_t<
    std::is_convertible_v<remove_decoration_t<typename std::iterator_traits<
                              InputIteratorT>::value_type>,
                          OutputT> &&
    is_group_helper_v<GroupHelper>>
group_load(GroupHelper gh, InputIteratorT in_ptr,
           span<OutputT, ElementsPerWorkItem> out, Properties properties) {
  constexpr bool blocked = detail::isBlocked(properties);

  using value_type = remove_decoration_t<
      typename std::iterator_traits<InputIteratorT>::value_type>;

  auto g = gh.get_group();
  using GroupTy = decltype(g);

  if constexpr (detail::no_shuffle_impl_available<
                    sizeof(value_type), ElementsPerWorkItem, blocked>) {
    return group_load(g, in_ptr, out, properties);
  } else {
    constexpr bool is_sg = detail::is_sub_group<GroupTy>::value;

    auto generic = [&]() {
      group_barrier(g);
      for (int i = 0; i < ElementsPerWorkItem; ++i)
        out[i] =
            in_ptr[detail::get_mem_idx<blocked, ElementsPerWorkItem>(g, i)];
      group_barrier(g);
    };

    auto delegate = [&](auto unwrapped_ptr) {
      group_load(gh, unwrapped_ptr, out, properties);
    };

    constexpr int BlockSize =
        sizeof(value_type) * (blocked ? ElementsPerWorkItem : 1);
    constexpr int NumBlocks = blocked ? 1 : ElementsPerWorkItem;

    constexpr auto hw_block_size = [&]() {
      size_t size = 8;
      while (BlockSize % size != 0)
        size /= 2;
      return size;
    }();

    using HWBlockTy = detail::cl_unsigned<hw_block_size>;

    constexpr const size_t hw_blocks_per_block = BlockSize / hw_block_size;

    auto impl = [&](auto *in_ptr) {
      auto sg = [&]() {
        if constexpr (is_sg)
          return g;
        else {
          // TODO: Use get_child_group from
          // sycl_ext_oneapi_root_group extension once it is
          // implemented instead of this free function.
          auto ndi = sycl::ext::oneapi::experimental::this_nd_item<
              GroupTy::dimensions>();
          return ndi.get_sub_group();
        }
      }();
      auto sg_lid = sg.get_local_linear_id();
      size_t sg_size = sg.get_max_local_range().size();
      size_t g_size = g.get_local_linear_range();

      group_barrier(g);
      auto scratch_span = gh.get_memory();
      // select next vec_size for the load.
      // 1 == 2^0,  16 == 2^4
      constexpr size_t max_vec_pwr_of_two = hw_block_size == 1 ? 4 : 3;

      size_t cur_hw_blocks_start_idx = 0;

      detail::loop<max_vec_pwr_of_two + 1>([&](auto i) {
        // Use bigger sizes first.
        constexpr int vec_size = 1 << (max_vec_pwr_of_two - i);

        constexpr auto iterations =
            i == 0
                ? hw_blocks_per_block * NumBlocks / vec_size
                : (hw_blocks_per_block * NumBlocks % (vec_size * 2)) / vec_size;
        detail::loop_unroll_up_to<iterations, 16>([&](auto) {
          const size_t hw_blocks_per_iter =
              g.get_local_linear_range() * vec_size;

          using LoadT = std::conditional_t<
              vec_size == 1, HWBlockTy,
              detail::ConvertToOpenCLType_t<vec<HWBlockTy, vec_size>>>;

          using PtrT = typename detail::DecoratedType<
              HWBlockTy, access::address_space::global_space>::type *;
          auto this_sg_offset = cur_hw_blocks_start_idx;
          if constexpr (!is_sg) {
            this_sg_offset += sg.get_group_id() * vec_size * sg_size;
          }
          LoadT load = __spirv_SubgroupBlockReadINTEL<LoadT>(
              reinterpret_cast<PtrT>(in_ptr) + this_sg_offset);

          detail::loop<vec_size>([&](auto idx) {
            // Operate in terms of a low-level "block" (HWBlockTy).
            auto mem_idx =
                this_sg_offset + idx * sg.get_local_linear_range() + sg_lid;

            auto BlockN = mem_idx / (g_size * hw_blocks_per_block);
            auto InBlockIdx = mem_idx % (g_size * hw_blocks_per_block);

            auto result_wi = InBlockIdx / hw_blocks_per_block;
            auto result_idx = InBlockIdx % hw_blocks_per_block;
            auto scratch_idx = result_wi * hw_blocks_per_block * NumBlocks +
                               BlockN * hw_blocks_per_block + result_idx;
            std::memcpy(scratch_span.data() + hw_block_size * scratch_idx,
                        reinterpret_cast<char *>(&load) + idx * hw_block_size,
                        hw_block_size);
          });

          cur_hw_blocks_start_idx += hw_blocks_per_iter;
        });
      });
      group_barrier(g);
      auto scratch_idx =
          g.get_local_linear_id() * sizeof(value_type) * ElementsPerWorkItem;
      std::memcpy(out.data(), scratch_span.data() + scratch_idx,
                  sizeof(value_type) * ElementsPerWorkItem);
      group_barrier(g);
    };

    constexpr bool assume_full_sg =
        properties.template has_property<property::full_sg_key>();
    detail::dispatch_ptr<4 /* read align in bytes */, assume_full_sg,
                         false /* unsupported */>(g, in_ptr, generic, delegate,
                                                  impl);
  }
}

// Store API span + group/sub_group overload.
template <typename GroupHelper, typename InputT,
          std::size_t ElementsPerWorkItem, typename OutputIteratorT,
          typename Properties = decltype(properties())>
std::enable_if_t<std::is_convertible_v<
                     InputT, remove_decoration_t<typename std::iterator_traits<
                                 OutputIteratorT>::value_type>> &&
                 detail::is_generic_group_v<GroupHelper>>
group_store(GroupHelper gh, const span<InputT, ElementsPerWorkItem> in,
            OutputIteratorT out_ptr, Properties properties = {}) {
  constexpr bool blocked = detail::isBlocked(properties);

  using value_type = remove_decoration_t<
      typename std::iterator_traits<OutputIteratorT>::value_type>;

  using GroupTy = GroupHelper;
  auto g = gh;

  auto generic = [&]() {
    group_barrier(g);
    for (int i = 0; i < in.size(); ++i)
      out_ptr[detail::get_mem_idx<blocked, ElementsPerWorkItem>(gh, i)] = in[i];
    group_barrier(g);
  };

  auto delegate = [&](auto unwrapped_ptr) {
    group_store(g, in, unwrapped_ptr, properties);
  };

  constexpr int BlockSize =
      sizeof(value_type) * (blocked ? ElementsPerWorkItem : 1);
  constexpr int NumBlocks = blocked ? 1 : ElementsPerWorkItem;

  constexpr auto hw_block_size = [&]() {
    size_t size = 8;
    while (BlockSize % size != 0)
      size /= 2;
    return size;
  }();
  using HWBlockTy = detail::cl_unsigned<hw_block_size>;

  constexpr const size_t hw_blocks_per_block = BlockSize / hw_block_size;

  auto impl_sg = [&](sub_group sg, auto *out_ptr) {
    value_type v[ElementsPerWorkItem];
    // Perform the required implicit conversion first.
    detail::loop_unroll_up_to<ElementsPerWorkItem, 16>(
        [&](size_t i) { v[i] = in[i]; });

    auto priv_ptr = reinterpret_cast<char *>(&v);

    // Needs to be 16 bytes aligned (4 for reads).

    // Native is strided!
    // Available native BlockSizes: uchar, ushort, uint, ulong (1, 2, 4, 8).
    // Available native NumBlocks:
    //   1, 2, 4, 8, or 16 uchars
    //   1, 2, 4, or 8 ulongs/uints/ushorts

    size_t sg_lid = sg.get_local_linear_id();
    size_t sg_size = sg.get_max_local_range().size(); // Assume "full" SG.

    size_t cur_hw_blocks_start_idx = 0;

    size_t cur_read_index = 0;

    // select next vec_size for the load.
    // 1 == 2^0,  16 == 2^4
    constexpr size_t max_vec_pwr_of_two = hw_block_size == 1 ? 4 : 3;
    detail::loop<max_vec_pwr_of_two + 1>([&](auto i) {
      // Use bigger sizes first.
      constexpr int vec_size = 1 << (max_vec_pwr_of_two - i);

      constexpr auto iterations =
          i == 0
              ? hw_blocks_per_block * NumBlocks / vec_size
              : (hw_blocks_per_block * NumBlocks % (vec_size * 2)) / vec_size;

      detail::loop_unroll_up_to<iterations, 16>([&](auto) {
        const size_t hw_blocks_per_iter = sg_size * vec_size;

        using StoreT = std::conditional_t<
            vec_size == 1, HWBlockTy,
            detail::ConvertToOpenCLType_t<vec<HWBlockTy, vec_size>>>;

        using PtrT = typename detail::DecoratedType<
            HWBlockTy, access::address_space::global_space>::type *;
        StoreT store_val;

        if constexpr (hw_blocks_per_block == 1) {
          std::memcpy(&store_val, priv_ptr + cur_read_index * hw_block_size,
                      sizeof(store_val));
          cur_read_index += vec_size;
        } else if constexpr (detail::is_power_of_two(hw_blocks_per_block) &&
                             vec_size >= hw_blocks_per_block) {
          //  Idx\WI |     0    |     1    |
          //    0    | s0.w0.b0 | s0.w0.b2 |
          //    1    | s0.w1.b0 | s0.w1.b2 |
          //    2    | s0.w0.b1 | s0.w0.b3 |
          //    3    | s0.w1.b1 | s0.w1.b3 |

          // reverse mapping:

          //  SG.Idx\WI |    0   |     1  |
          //  s0.b0     |  w0.0  |  w0.1  |
          //  s0.b1     |  w0.2  |  w0.3  |
          //  s0.b2     |  w1.0  |  w1.1  |
          //  s0.b3     |  w1.2  |  w1.3  |

          //  Idx\WI |     0    |     1    |
          // 0|.0    | s0.w0.b0 | s0.w0.b2 |
          //  |.1    | s0.w1.b0 | s0.w1.b2 |
          //  |.2    | s0.w0.b1 | s0.w0.b3 |
          //  |.3    | s0.w1.b1 | s0.w1.b3 |
          //         +----------+----------+
          // 1|.0    | s0.w0.b4 | s0.w0.b6 |
          //  |.1    | s0.w1.b4 | s0.w1.b6 |
          //  |.2    | s0.w0.b5 | s0.w0.b7 |
          //  |.3    | s0.w1.b5 | s0.w1.b7 |

          // reverse mapping:

          //  SG.Idx\WI |    0     |     1    |
          //  s0.b0     |  w0.0.0  |  w0.0.1  |
          //  s0.b1     |  w0.0.2  |  w0.0.3  |
          //  s0.b2     |  w1.0.0  |  w1.0.1  |
          //  s0.b3     |  w1.0.2  |  w1.0.3  |
          //  s0.b4     |  w0.1.0  |  w0.1.1  |
          //  s0.b5     |  w0.1.2  |  w0.1.3  |
          //  s0.b6     |  w1.1.0  |  w1.1.1  |
          //  s0.b7     |  w1.1.2  |  w1.1.3  |

          //  Idx\WI |     0    |     1    |     2    |     3    |
          //    0    | s0.w0.b0 | s0.w2.b0 | s0.w0.b1 | s0.w2.b1 |
          //    1    | s0.w1.b0 | s0.w3.b0 | s0.w1.b1 | s0.w3.b1 |

          // reverse mapping:
          //  SG.Idx\WI |   0   |  1   |  2   |  3   |
          //  s0.b0     | w0.0  | w0.1 | w1.0 | w1.1 |
          //  s0.b1     | w2.0  | w2.1 | w3.0 | w3.1 |

          detail::loop<vec_size>([&](auto i) {
            size_t idx = i * sg_size + sg_lid;

            size_t block_idx = idx % hw_blocks_per_block;
            size_t wi = (idx / hw_blocks_per_block) % sg_size;
            size_t BlockIdx = i / hw_blocks_per_block; // uniform

            HWBlockTy Block[hw_blocks_per_block];
            std::memcpy(&Block,
                        reinterpret_cast<char *>(v) + sizeof(Block) * BlockIdx,
                        sizeof(Block));

            HWBlockTy ShuffledBlock[hw_blocks_per_block];
            // TODO: Report a bug?
            // undefined reference to
            // `__builtin_spirv_OpSubgroupShuffleINTEL_v2i64_i32'
            detail::loop_unroll_up_to<hw_blocks_per_block, 16>([&](auto i) {
              ShuffledBlock[i] = select_from_group(sg, Block[i], wi);
            });
            HWBlockTy val = ShuffledBlock[block_idx];

            std::memcpy(reinterpret_cast<char *>(&store_val) +
                            i * hw_block_size,
                        &val, hw_block_size);
          });
        } else {
          // See "unsupported" below in detail::dispatch_ptr invocation.
          static_assert(hw_blocks_per_block == 0,
                        "Should have bailed out earlier!");
        }

        __spirv_SubgroupBlockWriteINTEL(reinterpret_cast<PtrT>(out_ptr) +
                                            cur_hw_blocks_start_idx,
                                        store_val);

        cur_hw_blocks_start_idx += hw_blocks_per_iter;
      });
    });
  };

  auto impl = [&](auto *out_ptr) {
    group_barrier(g);
    if constexpr (detail::is_sub_group<GroupTy>::value) {
      return impl_sg(g, out_ptr);
    } else {
      // TODO: Use get_child_group from sycl_ext_oneapi_root_group extension
      // once it is implemented instead of this free function.
      auto ndi =
          sycl::ext::oneapi::experimental::this_nd_item<GroupTy::dimensions>();
      auto sg = ndi.get_sub_group();
      if constexpr (blocked) {
        return impl_sg(sg, out_ptr + sg.get_group_id() *
                                         sg.get_max_local_range() *
                                         ElementsPerWorkItem);
      } else {
        // For striped layout the stride between elements in a vector is
        // expressed in terms of WG's size, not SG. As such, each index has
        // to be implemented using scalar SG block load.
        auto vec_elem_stride = g.get_local_linear_range();
        detail::loop_unroll_up_to<ElementsPerWorkItem, 16>([&](size_t i) {
          value_type scalar = in[i]; // implicit conversion.
          group_store(sg, scalar,
                      out_ptr + sg.get_group_id() * sg.get_max_local_range() +
                          vec_elem_stride * i,
                      properties);
        });
      }
    }
    group_barrier(g);
  };

  constexpr bool assume_full_sg =
      properties.template has_property<property::full_sg_key>();
  constexpr bool unsupported =
      !detail::is_power_of_two(hw_blocks_per_block) ||
      hw_blocks_per_block > 16 ||
      (hw_blocks_per_block == 16 && hw_block_size != 1);
  detail::dispatch_ptr<16 /* read align in bytes */, assume_full_sg,
                       unsupported>(g, out_ptr, generic, delegate, impl);
}

// Store API span + group_helper overload.
template <typename GroupHelper, typename InputT,
          std::size_t ElementsPerWorkItem, typename OutputIteratorT,
          typename Properties = decltype(properties())>
std::enable_if_t<std::is_convertible_v<
                     InputT, remove_decoration_t<typename std::iterator_traits<
                                 OutputIteratorT>::value_type>> &&
                 is_group_helper_v<GroupHelper>>
group_store(GroupHelper gh, const span<InputT, ElementsPerWorkItem> in,
            OutputIteratorT out_ptr, Properties properties = {}) {
  constexpr bool blocked = detail::isBlocked(properties);

  using value_type = remove_decoration_t<
      typename std::iterator_traits<OutputIteratorT>::value_type>;

  auto g = gh.get_group();
  using GroupTy = decltype(g);

  if constexpr (detail::no_shuffle_impl_available<
                    sizeof(value_type), ElementsPerWorkItem, blocked>) {
    return group_store(g, in, out_ptr, properties);
  } else {
    constexpr bool is_sg = detail::is_sub_group<GroupTy>::value;

    auto generic = [&]() {
      group_barrier(g);
      for (int i = 0; i < in.size(); ++i)
        out_ptr[detail::get_mem_idx<blocked, ElementsPerWorkItem>(g, i)] =
            in[i];
      group_barrier(g);
    };

    auto delegate = [&](auto unwrapped_ptr) {
      group_store(gh, in, unwrapped_ptr, properties);
    };

    constexpr int BlockSize =
        sizeof(value_type) * (blocked ? ElementsPerWorkItem : 1);
    constexpr int NumBlocks = blocked ? 1 : ElementsPerWorkItem;

    constexpr auto hw_block_size = [&]() {
      size_t size = 8;
      while (BlockSize % size != 0)
        size /= 2;
      return size;
    }();

    using HWBlockTy = detail::cl_unsigned<hw_block_size>;

    constexpr const size_t hw_blocks_per_block = BlockSize / hw_block_size;

    auto impl = [&](auto *out_ptr) {
      auto sg = [&]() {
        if constexpr (is_sg)
          return g;
        else {
          // TODO: Use get_child_group from
          // sycl_ext_oneapi_root_group extension once it is
          // implemented instead of this free function.
          auto ndi = sycl::ext::oneapi::experimental::this_nd_item<
              GroupTy::dimensions>();
          return ndi.get_sub_group();
        }
      }();
      auto sg_lid = sg.get_local_linear_id();
      auto g_lid = g.get_local_linear_id();
      size_t sg_size = sg.get_max_local_range().size();
      size_t g_size = g.get_local_linear_range();

      group_barrier(g);
      auto scratch_span = gh.get_memory();

      for (int elem = 0; elem < NumBlocks; ++elem) {
        for (int block = 0; block < hw_blocks_per_block; ++block) {
          auto total_order_idx =
              blocked ? block + elem * hw_blocks_per_block +
                            g_lid * hw_blocks_per_block * NumBlocks
                      : block + g_lid * hw_blocks_per_block +
                            elem * hw_blocks_per_block * g_size;
          std::memcpy(scratch_span.data() + hw_block_size * total_order_idx,
                      reinterpret_cast<char *>(&in[elem]) +
                          hw_block_size * block,
                      hw_block_size);
        }
      }
      group_barrier(g);
      // select next vec_size for the load.
      // 1 == 2^0,  16 == 2^4
      constexpr size_t max_vec_pwr_of_two = hw_block_size == 1 ? 4 : 3;

      size_t cur_hw_blocks_start_idx = 0;

      detail::loop<max_vec_pwr_of_two + 1>([&](auto i) {
        // Use bigger sizes first.
        constexpr int vec_size = 1 << (max_vec_pwr_of_two - i);

        constexpr auto iterations =
            i == 0
                ? hw_blocks_per_block * NumBlocks / vec_size
                : (hw_blocks_per_block * NumBlocks % (vec_size * 2)) / vec_size;
        detail::loop_unroll_up_to<iterations, 16>([&](auto) {
          const size_t hw_blocks_per_iter =
              g.get_local_linear_range() * vec_size;
          using StoreT = std::conditional_t<
              vec_size == 1, HWBlockTy,
              detail::ConvertToOpenCLType_t<vec<HWBlockTy, vec_size>>>;
          using PtrT = typename detail::DecoratedType<
              HWBlockTy, access::address_space::global_space>::type *;
          auto this_sg_offset = cur_hw_blocks_start_idx;
          if constexpr (!is_sg) {
            this_sg_offset += sg.get_group_id() * vec_size * sg_size;
          }

          StoreT tmp;
          for (int i = 0; i < vec_size; ++i) {
            std::memcpy(reinterpret_cast<char *>(&tmp) + i * hw_block_size,
                        scratch_span.data() +
                            (this_sg_offset + sg_lid + i * sg_size) *
                                hw_block_size,
                        hw_block_size);
          }

          __spirv_SubgroupBlockWriteINTEL(
              reinterpret_cast<PtrT>(out_ptr) + this_sg_offset, tmp);
          cur_hw_blocks_start_idx += hw_blocks_per_iter;
        });
      });

      group_barrier(g);
    };

    constexpr bool assume_full_sg =
        properties.template has_property<property::full_sg_key>();
    detail::dispatch_ptr<16 /* write align in bytes */, assume_full_sg,
                         false /* unsupported */>(g, out_ptr, generic, delegate,
                                                  impl);
  }
}
#else
template <typename... Args> void group_load(Args...) {
  throw sycl::exception(
      std::error_code(PI_ERROR_INVALID_DEVICE, sycl::sycl_category()),
      "Group loads/stores are not supported on host.");
}
template <typename... Args> void group_store(Args...) {
  throw sycl::exception(
      std::error_code(PI_ERROR_INVALID_DEVICE, sycl::sycl_category()),
      "Group loads/stores are not supported on host.");
}
#endif
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
