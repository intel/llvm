//==---- group_load_store.hpp --- SYCL extension for group loads/stores ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements sycl_ext_oneapi_group_load_store extension.

#pragma once

#include <sycl/ext/oneapi/experimental/annotated_ptr/annotated_ptr.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>
#include <sycl/group_barrier.hpp>
#include <sycl/sycl_span.hpp>

#include <cstring>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

enum class data_placement_enum { blocked, striped };

struct data_placement_key
    : detail::compile_time_property_key<detail::PropKind::DataPlacement> {
  template <data_placement_enum Placement>
  using value_t =
      property_value<data_placement_key,
                     // TODO: Extension uses data_placement_enum directly here.
                     std::integral_constant<int, static_cast<int>(Placement)>>;
};

template <data_placement_enum Placement>
inline constexpr data_placement_key::value_t<Placement> data_placement;

inline constexpr data_placement_key::value_t<data_placement_enum::blocked>
    data_placement_blocked;
inline constexpr data_placement_key::value_t<data_placement_enum::striped>
    data_placement_striped;

struct contiguous_memory_key
    : detail::compile_time_property_key<detail::PropKind::ContiguousMemory> {
  using value_t = property_value<contiguous_memory_key>;
};

inline constexpr contiguous_memory_key::value_t contiguous_memory;

struct full_group_key
    : detail::compile_time_property_key<detail::PropKind::FullGroup> {
  using value_t = property_value<full_group_key>;
};

inline constexpr full_group_key::value_t full_group;

namespace detail {
struct naive_key : detail::compile_time_property_key<detail::PropKind::Naive> {
  using value_t = property_value<naive_key>;
};
inline constexpr naive_key::value_t naive;

struct native_local_block_io_key
    : detail::compile_time_property_key<detail::PropKind::NativeLocalBlockIO> {
  using value_t = property_value<native_local_block_io_key>;
};
inline constexpr native_local_block_io_key::value_t native_local_block_io;

using namespace sycl::detail;
} // namespace detail

#ifdef __SYCL_DEVICE_ONLY__
namespace detail {
template <typename InputIteratorT, typename OutputElemT>
inline constexpr bool verify_load_types =
    std::is_same_v<
        typename std::iterator_traits<InputIteratorT>::iterator_category,
        std::random_access_iterator_tag> &&
    std::is_convertible_v<remove_decoration_t<typename std::iterator_traits<
                              InputIteratorT>::value_type>,
                          OutputElemT> &&
    std::is_trivially_copyable_v<remove_decoration_t<
        typename std::iterator_traits<InputIteratorT>::value_type>> &&
    std::is_default_constructible_v<remove_decoration_t<
        typename std::iterator_traits<InputIteratorT>::value_type>> &&
    std::is_trivially_copyable_v<OutputElemT> &&
    std::is_default_constructible_v<OutputElemT>;

template <typename InputElemT, typename OutputIteratorT>
inline constexpr bool verify_store_types =
    std::is_same_v<
        typename std::iterator_traits<OutputIteratorT>::iterator_category,
        std::random_access_iterator_tag> &&
    std::is_convertible_v<InputElemT,
                          remove_decoration_t<typename std::iterator_traits<
                              OutputIteratorT>::value_type>> &&
    std::is_trivially_copyable_v<remove_decoration_t<
        typename std::iterator_traits<OutputIteratorT>::value_type>> &&
    std::is_default_constructible_v<remove_decoration_t<
        typename std::iterator_traits<OutputIteratorT>::value_type>> &&
    std::is_trivially_copyable_v<InputElemT> &&
    std::is_default_constructible_v<InputElemT>;

template <typename Properties> constexpr bool isBlocked(Properties properties) {
  if constexpr (properties.template has_property<data_placement_key>())
    return properties.template get_property<data_placement_key>() ==
           data_placement_blocked;
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

// SPIR-V extension:
// https://github.com/KhronosGroup/SPIRV-Registry/blob/main/extensions/INTEL/SPV_INTEL_subgroups.asciidoc,
// however it doesn't describe limitations/requirements. Those seem to be
// listed in the Intel OpenCL extensions for sub-groups:
// https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroups.html
// https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroups_char.html
// https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroups_long.html
// https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroups_short.html
// https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_local_block_io.html
// Reads require 4-byte alignment for global pointers and 16-byte alignment for
// local pointers, writes require 16-byte alignment. Supported sizes:
//
// +------------+-------------+
// | block type | # of blocks |
// +------------+-------------+
// | uchar      | 1,2,4,8,16  |
// | ushort     | 1,2,4,8,16  |
// | uint       | 1,2,4,8     |
// | ulong      | 1,2,4,8     |
// +------------+-------------+
//
// Utility type traits below are used to map user type to one of the block
// read/write types above.

template <typename IteratorT, std::size_t ElementsPerWorkItem, bool Blocked>
struct BlockInfo {
  using value_type =
      remove_decoration_t<typename std::iterator_traits<IteratorT>::value_type>;

  static constexpr int block_size =
      sizeof(value_type) * (Blocked ? ElementsPerWorkItem : 1);
  static constexpr int num_blocks = Blocked ? 1 : ElementsPerWorkItem;
  // There is an overload in the table above that could be used for the block
  // operation:
  static constexpr bool has_builtin =
      detail::is_power_of_two(block_size) &&
      detail::is_power_of_two(num_blocks) && block_size <= 8 &&
      (num_blocks <= 8 || (num_blocks == 16 && block_size <= 2));
};

enum class operation_type { load, store };

template <operation_type OpType, access::address_space Space>
struct RequiredAlignment {};

template <operation_type OpType>
struct RequiredAlignment<OpType, access::address_space::global_space> {
  static constexpr int value = (OpType == operation_type::load) ? 4 : 16;
};

template <operation_type OpType>
struct RequiredAlignment<OpType, access::address_space::local_space> {
  static constexpr int value = 16;
};

template <typename BlockInfoTy> struct BlockTypeInfo;

template <typename IteratorT, std::size_t ElementsPerWorkItem, bool Blocked>
struct BlockTypeInfo<BlockInfo<IteratorT, ElementsPerWorkItem, Blocked>> {
  using BlockInfoTy = BlockInfo<IteratorT, ElementsPerWorkItem, Blocked>;

  using block_type = detail::fixed_width_unsigned<BlockInfoTy::block_size>;

  using block_pointer_elem_type = std::conditional_t<
      std::is_const_v<std::remove_reference_t<
          typename std::iterator_traits<IteratorT>::reference>>,
      std::add_const_t<block_type>, block_type>;

  static constexpr auto deduced_address_space =
      detail::deduce_AS<std::remove_cv_t<IteratorT>>::value;

  using block_pointer_type =
      typename detail::DecoratedType<block_pointer_elem_type,
                                     deduced_address_space>::type *;

  using block_op_type = std::conditional_t<
      BlockInfoTy::num_blocks == 1, block_type,
      detail::ConvertToOpenCLType_t<vec<block_type, BlockInfoTy::num_blocks>>>;
};

// Returns either a pointer decorated with the deduced address space, suitable
// to use in a block read/write builtin, or nullptr if some legality conditions
// aren't satisfied. If deduced address space is generic then returned pointer
// will have generic address space and has to be dynamically casted to global or
// local space before using in a builtin.
template <std::size_t ElementsPerWorkItem, typename IteratorT,
          typename Properties>
constexpr auto get_block_op_ptr(IteratorT iter,
                                [[maybe_unused]] Properties props) {
  using iter_no_cv = std::remove_cv_t<IteratorT>;

  constexpr bool blocked = detail::isBlocked(props);
  using BlkInfo = BlockInfo<IteratorT, ElementsPerWorkItem, blocked>;

#if defined(__SPIR__)
  // TODO: What about non-Intel SPIR-V devices?
  constexpr bool is_spir = true;
#else
  constexpr bool is_spir = false;
#endif

  if constexpr (!is_spir || !BlkInfo::has_builtin) {
    return nullptr;
  } else if constexpr (!props.template has_property<full_group_key>()) {
    return nullptr;
  } else if constexpr (detail::is_multi_ptr_v<IteratorT>) {
    return get_block_op_ptr<ElementsPerWorkItem>(iter.get_decorated(), props);
  } else if constexpr (!std::is_pointer_v<iter_no_cv>) {
    if constexpr (props.template has_property<contiguous_memory_key>())
      return get_block_op_ptr<ElementsPerWorkItem>(&*iter, props);
    else
      return nullptr;
  } else {
    // Load/store to/from nullptr would be an UB, this assume allows the
    // compiler to optimize the IR further.
    __builtin_assume(iter != nullptr);

    using block_pointer_type =
        typename BlockTypeInfo<BlkInfo>::block_pointer_type;

    constexpr auto deduced_address_space =
        BlockTypeInfo<BlkInfo>::deduced_address_space;

    if constexpr (deduced_address_space ==
                      access::address_space::generic_space ||
                  deduced_address_space ==
                      access::address_space::global_space ||
                  (deduced_address_space ==
                       access::address_space::local_space &&
                   props.template has_property<
                       detail::native_local_block_io_key>())) {
      return reinterpret_cast<block_pointer_type>(iter);
    } else {
      return nullptr;
    }
  }
}

template <int RequiredAlign, typename IteratorType, typename Properties>
bool is_aligned(IteratorType iter, [[maybe_unused]] Properties props) {
  using value_type = remove_decoration_t<
      typename std::iterator_traits<IteratorType>::value_type>;

  if constexpr (Properties::template has_property<alignment_key>()) {
    if (Properties::template get_property<alignment_key>().value >=
        RequiredAlign)
      return true;
  }

  return alignof(value_type) >= RequiredAlign ||
         reinterpret_cast<uintptr_t>(&*iter) % RequiredAlign == 0;
}

template <typename Group, typename InputIteratorT, typename OutputT,
          std::size_t ElementsPerWorkItem,
          typename Properties = decltype(properties())>
std::enable_if_t<detail::verify_load_types<InputIteratorT, OutputT> &&
                 detail::is_generic_group_v<Group> &&
                 is_property_list_v<Properties>>
group_load_impl(Group g, InputIteratorT in_ptr,
                span<OutputT, ElementsPerWorkItem> out, Properties props = {}) {
  constexpr bool blocked = detail::isBlocked(props);
  using use_naive =
      detail::merged_properties_t<Properties,
                                  decltype(properties(detail::naive))>;

  if constexpr (props.template has_property<detail::naive_key>()) {
    group_barrier(g);
    for (int i = 0; i < out.size(); ++i)
      out[i] = in_ptr[detail::get_mem_idx<blocked, ElementsPerWorkItem>(g, i)];
    group_barrier(g);
    return;
  } else if constexpr (!std::is_same_v<Group, sycl::sub_group>) {
    return group_load_impl(g, in_ptr, out, use_naive{});
  } else {
    auto ptr = detail::get_block_op_ptr<ElementsPerWorkItem>(in_ptr, props);
    static constexpr auto deduced_address_space =
        detail::deduce_AS<std::remove_cv_t<decltype(ptr)>>::value;

    if constexpr (!std::is_same_v<std::nullptr_t, decltype(ptr)>) {
      if constexpr (deduced_address_space ==
                    access::address_space::generic_space) {
        if (auto local_ptr = detail::dynamic_address_cast<
                access::address_space::local_space>(ptr)) {
          return group_load_impl(g, local_ptr, out, props);
        } else if (auto global_ptr = detail::dynamic_address_cast<
                       access::address_space::global_space>(ptr)) {
          return group_load_impl(g, global_ptr, out, props);
        } else {
          return group_load_impl(g, in_ptr, out, use_naive{});
        }
      } else {
        using value_type = remove_decoration_t<
            typename std::iterator_traits<InputIteratorT>::value_type>;
        using block_info = typename detail::BlockTypeInfo<
            detail::BlockInfo<InputIteratorT, ElementsPerWorkItem, blocked>>;
        using block_op_type = typename block_info::block_op_type;
        // Alignment checks of the pointer.
        constexpr int ReqAlign =
            detail::RequiredAlignment<detail::operation_type::load,
                                      deduced_address_space>::value;
        if (!detail::is_aligned<ReqAlign>(in_ptr, props))
          return group_load_impl(g, in_ptr, out, use_naive{});

        // We know the pointer is aligned and the address space is known. Do the
        // optimized load.
        auto load = __spirv_SubgroupBlockReadINTEL<block_op_type>(ptr);

        // TODO: accessor_iterator's value_type is weird, so we need
        // `std::remove_const_t` below:
        //
        // static_assert(
        //     std::is_same_v<
        //         typename std::iterator_traits<
        //             sycl::detail::accessor_iterator<const int,
        //             1>>::value_type,
        //         const int>);
        //
        // yet
        //
        // static_assert(
        //     std::is_same_v<
        //         typename std::iterator_traits<const int *>::value_type,
        //         int>);
        if constexpr (std::is_same_v<std::remove_const_t<value_type>,
                                     OutputT>) {
          static_assert(sizeof(load) == out.size_bytes());
          sycl::detail::memcpy_no_adl(out.begin(), &load, out.size_bytes());
        } else {
          std::remove_const_t<value_type> values[ElementsPerWorkItem];
          static_assert(sizeof(load) == sizeof(values));
          sycl::detail::memcpy_no_adl(values, &load, sizeof(values));

          // Note: can't `memcpy` directly into `out` because that might bypass
          // an implicit conversion required by the specification.
          for (int i = 0; i < ElementsPerWorkItem; ++i)
            out[i] = values[i];
        }
      }
    } else {
      return group_load_impl(g, in_ptr, out, use_naive{});
    }

    return;
  }
}

template <typename Group, typename InputT, std::size_t ElementsPerWorkItem,
          typename OutputIteratorT,
          typename Properties = decltype(properties())>
std::enable_if_t<detail::verify_store_types<InputT, OutputIteratorT> &&
                 detail::is_generic_group_v<Group> &&
                 is_property_list_v<Properties>>
group_store_impl(Group g, const span<InputT, ElementsPerWorkItem> in,
                 OutputIteratorT out_ptr, Properties props = {}) {
  constexpr bool blocked = detail::isBlocked(props);
  using use_naive =
      detail::merged_properties_t<Properties,
                                  decltype(properties(detail::naive))>;

  if constexpr (props.template has_property<detail::naive_key>()) {
    group_barrier(g);
    for (int i = 0; i < in.size(); ++i)
      out_ptr[detail::get_mem_idx<blocked, ElementsPerWorkItem>(g, i)] = in[i];
    group_barrier(g);
    return;
  } else if constexpr (!std::is_same_v<Group, sycl::sub_group>) {
    return group_store_impl(g, in, out_ptr, use_naive{});
  } else {
    auto ptr = detail::get_block_op_ptr<ElementsPerWorkItem>(out_ptr, props);

    if constexpr (!std::is_same_v<std::nullptr_t, decltype(ptr)>) {
      static constexpr auto deduced_address_space =
          detail::deduce_AS<std::remove_cv_t<decltype(ptr)>>::value;
      if constexpr (deduced_address_space ==
                    access::address_space::generic_space) {
        if (auto local_ptr = detail::dynamic_address_cast<
                access::address_space::local_space>(ptr)) {
          return group_store_impl(g, in, local_ptr, props);
        } else if (auto global_ptr = detail::dynamic_address_cast<
                       access::address_space::global_space>(ptr)) {
          return group_store_impl(g, in, global_ptr, props);
        } else {
          return group_store_impl(g, in, out_ptr, use_naive{});
        }
      } else {
        using block_info = typename detail::BlockTypeInfo<
            detail::BlockInfo<OutputIteratorT, ElementsPerWorkItem, blocked>>;
        using block_op_type = typename block_info::block_op_type;

        // Alignment checks of the pointer.
        constexpr int ReqAlign =
            detail::RequiredAlignment<detail::operation_type::store,
                                      deduced_address_space>::value;
        if (!detail::is_aligned<ReqAlign>(out_ptr, props))
          return group_store_impl(g, in, out_ptr, use_naive{});

        std::remove_const_t<remove_decoration_t<
            typename std::iterator_traits<OutputIteratorT>::value_type>>
            values[ElementsPerWorkItem];

        for (int i = 0; i < ElementsPerWorkItem; ++i) {
          // Including implicit conversion.
          values[i] = in[i];
        }

        // We know the pointer is aligned and the address space is known. Do the
        // optimized load.
        __spirv_SubgroupBlockWriteINTEL(ptr,
                                        sycl::bit_cast<block_op_type>(values));
      }
    } else {
      return group_store_impl(g, in, out_ptr, use_naive{});
    }
  }
}
} // namespace detail

// Load API span overload.
template <typename Group, typename InputIteratorT, typename OutputT,
          std::size_t ElementsPerWorkItem,
          typename Properties = decltype(properties())>
std::enable_if_t<detail::verify_load_types<InputIteratorT, OutputT> &&
                 detail::is_generic_group_v<Group> &&
                 is_property_list_v<Properties>>
group_load(Group g, InputIteratorT in_ptr,
           span<OutputT, ElementsPerWorkItem> out, Properties props = {}) {
  static_assert(std::is_pointer_v<InputIteratorT> ||
                    !Properties::template has_property<alignment_key>(),
                "group_load requires a pointer if alignment property is used");
  detail::group_load_impl(g, in_ptr, out, props);
}

// Store API span overload.
template <typename Group, typename InputT, std::size_t ElementsPerWorkItem,
          typename OutputIteratorT,
          typename Properties = decltype(properties())>
std::enable_if_t<detail::verify_store_types<InputT, OutputIteratorT> &&
                 detail::is_generic_group_v<Group> &&
                 is_property_list_v<Properties>>
group_store(Group g, const span<InputT, ElementsPerWorkItem> in,
            OutputIteratorT out_ptr, Properties props = {}) {
  static_assert(std::is_pointer_v<OutputIteratorT> ||
                    !Properties::template has_property<alignment_key>(),
                "group_store requires a pointer if alignment property is used");
  detail::group_store_impl(g, in, out_ptr, props);
}

// Load API scalar.
template <typename Group, typename InputIteratorT, typename OutputT,
          typename Properties = decltype(properties())>
std::enable_if_t<detail::verify_load_types<InputIteratorT, OutputT> &&
                 detail::is_generic_group_v<Group> &&
                 is_property_list_v<Properties>>
group_load(Group g, InputIteratorT in_ptr, OutputT &out,
           Properties properties = {}) {
  static_assert(std::is_pointer_v<InputIteratorT> ||
                    !Properties::template has_property<alignment_key>(),
                "group_load requires a pointer if alignment property is used");
  detail::group_load_impl(g, in_ptr, span<OutputT, 1>(&out, 1), properties);
}

// Store API scalar.
template <typename Group, typename InputT, typename OutputIteratorT,
          typename Properties = decltype(properties())>
std::enable_if_t<detail::verify_store_types<InputT, OutputIteratorT> &&
                 detail::is_generic_group_v<Group> &&
                 is_property_list_v<Properties>>
group_store(Group g, const InputT &in, OutputIteratorT out_ptr,
            Properties properties = {}) {
  static_assert(std::is_pointer_v<OutputIteratorT> ||
                    !Properties::template has_property<alignment_key>(),
                "group_store requires a pointer if alignment property is used");
  detail::group_store_impl(g, span<const InputT, 1>(&in, 1), out_ptr,
                           properties);
}

// Load API sycl::vec overload.
template <typename Group, typename InputIteratorT, typename OutputT, int N,
          typename Properties = decltype(properties())>
std::enable_if_t<detail::verify_load_types<InputIteratorT, OutputT> &&
                 detail::is_generic_group_v<Group> &&
                 is_property_list_v<Properties>>
group_load(Group g, InputIteratorT in_ptr, sycl::vec<OutputT, N> &out,
           Properties properties = {}) {
  static_assert(std::is_pointer_v<InputIteratorT> ||
                    !Properties::template has_property<alignment_key>(),
                "group_load requires a pointer if alignment property is used");
  detail::group_load_impl(g, in_ptr, span<OutputT, N>(&out[0], N), properties);
}

// Store API sycl::vec overload.
template <typename Group, typename InputT, int N, typename OutputIteratorT,
          typename Properties = decltype(properties())>
std::enable_if_t<detail::verify_store_types<InputT, OutputIteratorT> &&
                 detail::is_generic_group_v<Group> &&
                 is_property_list_v<Properties>>
group_store(Group g, const sycl::vec<InputT, N> &in, OutputIteratorT out_ptr,
            Properties properties = {}) {
  static_assert(std::is_pointer_v<OutputIteratorT> ||
                    !Properties::template has_property<alignment_key>(),
                "group_store requires a pointer if alignment property is used");
  detail::group_store_impl(g, span<const InputT, N>(&in[0], N), out_ptr,
                           properties);
}

#else
template <typename... Args> void group_load(Args...) {
  throw sycl::exception(
      sycl::errc::feature_not_supported,
      "Group loads/stores are not supported on host.");
}
template <typename... Args> void group_store(Args...) {
  throw sycl::exception(
      sycl::errc::feature_not_supported,
      "Group loads/stores are not supported on host.");
}
#endif
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
