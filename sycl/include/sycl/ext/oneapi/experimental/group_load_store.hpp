// Implements https://github.com/intel/llvm/pull/7593

#pragma once

#include <sycl/ext/oneapi/properties/properties.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
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
} // namespace property

template <>
struct is_property_key<property::data_placement_key> : std::true_type {};

namespace detail {
template <> struct PropertyToKind<property::data_placement_key> {
  static constexpr PropKind Kind = PropKind::DataPlacement;
};
template <>
struct IsCompileTimeProperty<property::data_placement_key> : std::true_type {};

template <typename Properties>
constexpr bool is_blocked(Properties properties) {
  if constexpr (properties
                    .template has_property<property::data_placement_key>())
    return properties.template get_property<property::data_placement_key>() ==
           property::data_placement<group_algorithm_data_placement::blocked>;
  else
    return true;
}

template <bool IsBlocked, int VEC_OR_ARRAY_SIZE, typename GroupHelper>
int get_mem_idx(GroupHelper gh, int vec_or_array_idx) {
  if constexpr (IsBlocked)
    return gh.get_local_linear_id() * VEC_OR_ARRAY_SIZE + vec_or_array_idx;
  else
    return gh.get_local_linear_id() +
           gh.get_local_linear_range() * vec_or_array_idx;
}

template <typename T> struct is_multi_ptr_impl : public std::false_type {};

template <typename T, access::address_space Space,
          access::decorated DecorateAddress>
struct is_multi_ptr_impl<multi_ptr<T, Space, DecorateAddress>>
    : public std::true_type {};

template <typename T>
constexpr bool is_multi_ptr_v = is_multi_ptr_impl<std::remove_cv_t<T>>::value;

#if defined(__SPIR__)
constexpr bool is_spir = true;
#else
constexpr bool is_spir = false;
#endif

#ifdef __SYCL_DEVICE_ONLY__
template <typename ElemTy>
auto cast_ptr_for_block_op(ElemTy *ptr) {
  using BlockT = sycl::detail::sub_group::SelectBlockT<ElemTy>;
  using PtrT =
      sycl::detail::DecoratedType<BlockT,
                                  access::address_space::global_space>::type *;
  return reinterpret_cast<PtrT>(ptr);
}

template <int N, typename ElemTy> auto block_read(ElemTy *ptr) {
  using UndecoratedT = remove_decoration_t<ElemTy>;
  using BlockT = sycl::detail::sub_group::SelectBlockT<ElemTy>;
  using LoadT =
      // TODO: Can we use UndecoratedT instead of BlockT here?
      std::conditional_t<N == 1, BlockT,
                         sycl::detail::ConvertToOpenCLType_t<vec<BlockT, N>>>;
  LoadT load =
      __spirv_SubgroupBlockReadINTEL<LoadT>(cast_ptr_for_block_op(ptr));

  using RetT = std::conditional_t<N == 1, UndecoratedT, vec<UndecoratedT, N>>;

  return sycl::bit_cast<RetT>(load);
}

template <typename ElemTy>
void block_write(ElemTy *ptr, remove_decoration_t<ElemTy> val) {
  using BlockT = sycl::detail::sub_group::SelectBlockT<ElemTy>;
  __spirv_SubgroupBlockWriteINTEL(cast_ptr_for_block_op(ptr),
                                  sycl::bit_cast<BlockT>(val));
}

template <typename ElemTy, int N>
void block_write(ElemTy *ptr, vec<remove_decoration_t<ElemTy>, N> val) {
  using BlockT = sycl::detail::sub_group::SelectBlockT<ElemTy>;
  using VecT = sycl::detail::ConvertToOpenCLType_t<vec<BlockT, N>>;
  __spirv_SubgroupBlockWriteINTEL(cast_ptr_for_block_op(ptr),
                                  sycl::bit_cast<VecT>(val));
}
#endif

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
// - "delegate" WG to SG. This requires cooperation from the "delegate" callback
//   function as the stride might be a multiple of WG size.
// - finally, if the pointer is in the global address space and satisfies
//   alignment conditions, use "impl" callback to perform optimized load/store.
//
// Note that the tests for the functionality assume existence of this helper to
// avoid combinatorial explosion of scenarios to test.
template <int required_align, int sg_offset_per_wi, bool unsupported = false,
          typename GroupTy, typename IteratorT, typename GenericTy,
          typename DelegateTy, typename ImplTy>
void dispatch(GroupTy group, IteratorT iter, GenericTy generic,
              DelegateTy delegate, ImplTy impl) {
#ifdef __SYCL_DEVICE_ONLY__
  using value_type = std::iterator_traits<IteratorT>::value_type;
  using iter_no_cv = std::remove_cv_t<IteratorT>;
  auto generic_with_barrier = [&]() {
    group_barrier(group);
    generic();
    group_barrier(group);
  };
  if constexpr (!detail::is_spir || unsupported) {
    return generic_with_barrier();
  } else if constexpr (detail::is_multi_ptr_v<IteratorT>) {
    return delegate(group, iter.get_decorated());
  } else if constexpr (!std::is_pointer_v<iter_no_cv>) {
    return generic_with_barrier();
  } else {
    // Pointer.
    if constexpr (alignof(value_type) < required_align) {
      if ((reinterpret_cast<uintptr_t>(iter) % required_align) != 0) {
        return generic_with_barrier();
      }
    }

    // TODO: support scratchpad.
    if constexpr (std::is_same_v<GroupTy, sub_group>) {
      if (group.get_local_range() != group.get_max_local_range())
        // Sub-group is not "full".
        return generic_with_barrier();

      constexpr auto AS = sycl::detail::deduce_AS<iter_no_cv>::value;
      if constexpr (AS == access::address_space::global_space) {
        // The only customization point - to be handled by the
        // caller.
        return impl(iter);
      } else if constexpr (AS == access::address_space::generic_space) {
        if (auto global_ptr = __SYCL_GenericCastToPtrExplicit_ToGlobal<
                remove_decoration_t<value_type>>(iter))
          return delegate(group, global_ptr);
        else
          return generic_with_barrier();
      } else {
        return generic_with_barrier();
      }
    } else {
      // TODO: Use get_child_group from
      // sycl_ext_oneapi_root_group extension once it is
      // implemented instead of this free function.
      auto ndi =
          sycl::ext::oneapi::experimental::this_nd_item<GroupTy::dimensions>();
      auto sg = ndi.get_sub_group();

      auto wg_size = group.get_local_range().size();
      auto simd_width = sg.get_max_local_range().size();

      if (wg_size % simd_width != 0) {
        // TODO: Mapping to sub_group is implementation-defined,
        // no generic implementation is possible.
        return generic_with_barrier();
      } else {
        group_barrier(group);
        // TODO: verify that sizeof * sg.get_max_local_range() *
        // sg_offset_per_wi is a multiple of required alignment.
        delegate(sg, iter + sg.get_group_id() *
                                       sg.get_max_local_range() *
                                       sg_offset_per_wi);
        group_barrier(group);
        return;
      }
    }
  }
  __builtin_unreachable();
#endif
}
} // namespace detail

// Load API scalar
template <typename Group, typename InputIteratorT, typename OutputT,
          typename Properties = decltype(properties()),
          typename = std::enable_if_t<std::is_convertible_v<
              remove_decoration_t<
                  typename std::iterator_traits<InputIteratorT>::value_type>,
              OutputT>>>
void group_load(Group g, InputIteratorT in_ptr, OutputT &out,
                Properties properties = {}) {
#ifdef __SYCL_DEVICE_ONLY__
  using value_type = std::iterator_traits<InputIteratorT>::value_type;

  constexpr auto size = sizeof(value_type);
  constexpr bool supported_size =
      size == 1 || size == 2 || size == 4 || size == 8;
  constexpr bool unsupported = !supported_size;

  detail::dispatch<4 /* read align in bytes */, 1 /* scalar */, unsupported>(
      g, in_ptr, [&]() { out = in_ptr[g.get_local_linear_id()]; },
      [&](auto g, auto unwrapped_ptr) {
        group_load(g, unwrapped_ptr, out, properties);
      },
      [&](auto *in_ptr) { out = detail::block_read<1>(in_ptr); });
#else
  throw sycl::exception(
      std::error_code(PI_ERROR_INVALID_DEVICE, sycl::sycl_category()),
      "Group loads/stoes are not supported on host.");
#endif
}

// Store API scalar
template <typename Group, typename InputT, typename OutputIteratorT,
          typename Properties = decltype(properties()),
          typename = std::enable_if_t<std::is_convertible_v<
              InputT, remove_decoration_t<typename std::iterator_traits<
                          OutputIteratorT>::value_type>>>>
void group_store(Group g, const InputT &in, OutputIteratorT out_ptr,
                 Properties properties = {}) {
#ifdef __SYCL_DEVICE_ONLY__
  using value_type = std::iterator_traits<OutputIteratorT>::value_type;
  constexpr auto size = sizeof(value_type);
  constexpr bool supported_size =
      size == 1 || size == 2 || size == 4 || size == 8;
  constexpr bool unsupported = !supported_size;

  detail::dispatch<16 /* write align in bytes */, 1 /* scalar */, unsupported>(
      g, out_ptr, [&]() { out_ptr[g.get_local_linear_id()] = in; },
      [&](auto g, auto unwrapped_ptr) {
        group_store(g, in, unwrapped_ptr, properties);
      },
      [&](auto *out_ptr) {
        // FIXME: This is probably wrong as doesn't perform implicit conversion.
        detail::block_write(out_ptr, in);
      });
#else
  throw sycl::exception(
      std::error_code(PI_ERROR_INVALID_DEVICE, sycl::sycl_category()),
      "Group loads/stoes are not supported on host.");
#endif
}

// Load API sycl::vec overload
template <typename Group, typename InputIteratorT, typename OutputT, int N,
          typename Properties = decltype(properties()),
          typename = std::enable_if_t<std::is_convertible_v<
              remove_decoration_t<
                  typename std::iterator_traits<InputIteratorT>::value_type>,
              OutputT>>>
void group_load(Group g, InputIteratorT in_ptr, sycl::vec<OutputT, N> &out,
                Properties properties = {}) {
#ifdef __SYCL_DEVICE_ONLY__
  constexpr bool blocked = detail::is_blocked(properties);

  using value_type = std::iterator_traits<InputIteratorT>::value_type;
  constexpr auto size = sizeof(value_type);

  constexpr bool supported_size =
      size == 1 || size == 2 || size == 4 || size == 8;
  constexpr bool unsupported_vec_size = (N == 3 || (N == 16 && size != 1));
  constexpr bool unsupported = !supported_size || unsupported_vec_size;

  constexpr auto sg_offset_per_wi = blocked ? N : 1;

  detail::dispatch<4 /* read align in bytes */, sg_offset_per_wi, unsupported>(
      g, in_ptr,
      [&]() {
        sycl::detail::dim_loop<N>([&](size_t i) {
          out[i] = in_ptr[detail::get_mem_idx<blocked, N>(g, i)];
        });
      },
      [&](auto dispatch_g, auto unwrapped_ptr) {
        if constexpr (std::is_same_v<decltype(dispatch_g), Group> || blocked) {
          group_load(dispatch_g, unwrapped_ptr, out, properties);
        } else {
          // For striped layout the stride between elements in a vector is
          // expressed in terms of WG's size, not SG. As such, each index has
          // to be implemented using scalar SG block load.
          auto vec_elem_stride = g.get_local_linear_range();
          sycl::detail::dim_loop<N>([&](size_t i) {
            OutputT scalar;
            group_load(dispatch_g, unwrapped_ptr + vec_elem_stride * i, scalar,
                       properties);
            out[i] = scalar;
          });
        }
      },
      [&](auto *in_ptr) {
        // Make sure in_ptr is "auto" so that the body isn't "compiled" when
        // not instantiated due to "if constexpr".
        auto tmp = detail::block_read<N>(in_ptr);
        // SPIR-V builtin assumes striped layout.
        if constexpr (blocked) {
          // clang-format off
          // Data         |  0  1  2 |  3  4  5 |  6  7  8 |  9 10 11 | 12 13 14 | 15 16 17 | 18 19 20 | 21 22 23 |
          //
          // Was loaded as
          // vec_idx\WI   |  0 |  1 |  2 |  3 |  4 |  5 |  6 |  7
          //              |  8 |  9 | 10 | 11 | 12 | 13 | 14 | 15
          //              | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23
          //
          // Need to shuffle as
          //                  0        1       2       3      4       5       6       7
          //               v[0](0)  v[0](3) v[0](6) v[1](1) v[1](4) v[1](7) v[2](2) v[2](5)
          //               v[0](1)  v[0](4) v[0](7) v[1](2) v[1](5) v[2](0) v[2](3) v[2](6)
          //               v[0](2)  v[0](5) v[1](0) v[1](3) v[1](6) v[2](1) v[2](4) v[2](7)
          //
          // clang-format on

          vec<OutputT, N> shuffled = {0, 0};
          auto lid = g.get_local_id();
          int sg_size = g.get_max_local_range().size();
          sycl::detail::dim_loop<N>([&](size_t i) {
            // IMPORTANT: Note that the index for the v[y] depends on the
            // *CURRENT* WI, so the access into the vector must be done
            // *after* the shuffle.
            shuffled[i] = select_from_group(
                g, tmp, (lid * N + i) % sg_size)[(lid * N + i) / sg_size];
          });
          out = shuffled; // implicit conversion here.
        } else {
          out = tmp; // implicit conversion here.
        }
      });
#else
  throw sycl::exception(
      std::error_code(PI_ERROR_INVALID_DEVICE, sycl::sycl_category()),
      "Group loads/stoes are not supported on host.");
#endif
}

// Store API sycl::vec overload
template <typename Group, typename InputT, int N, typename OutputIteratorT,
          typename Properties = decltype(properties()),
          typename = std::enable_if_t<std::is_convertible_v<
              InputT, remove_decoration_t<typename std::iterator_traits<
                          OutputIteratorT>::value_type>>>>
void group_store(Group g, const sycl::vec<InputT, N> &in,
                 OutputIteratorT out_ptr, Properties properties = {}) {
  // TODO: Consider delegating to sycl::span version via sycl::bit_cast.
  constexpr bool blocked = detail::is_blocked(properties);
  auto generic = [&]() {
    sycl::detail::dim_loop<N>([&](size_t i) {
      out_ptr[detail::get_mem_idx<blocked, N>(g, i)] = in[i];
    });
  };

  group_barrier(g);
  generic();
  group_barrier(g);
}

// Load API sycl::span overload
template <typename GroupHelper, typename InputIteratorT, typename OutputT,
          std::size_t ElementsPerWorkItem,
          typename Properties = decltype(properties()),
          typename = std::enable_if_t<std::is_convertible_v<
              remove_decoration_t<
                  typename std::iterator_traits<InputIteratorT>::value_type>,
              OutputT>>>
void group_load(GroupHelper gh, InputIteratorT in_ptr,
                sycl::span<OutputT, ElementsPerWorkItem> &out,
                Properties properties = {}) {
  constexpr bool blocked = detail::is_blocked(properties);
  auto generic = [&]() {
    for (int i = 0; i < out.size(); ++i)
      out[i] = in_ptr[detail::get_mem_idx<blocked, ElementsPerWorkItem>(gh, i)];
  };

  group_barrier(gh);
  generic();
  group_barrier(gh);
}

// Store API sycl::span overload
template <typename GroupHelper, typename InputT,
          std::size_t ElementsPerWorkItem, typename OutputIteratorT,
          typename Properties = decltype(properties()),
          typename = std::enable_if_t<std::is_convertible_v<
              InputT, remove_decoration_t<typename std::iterator_traits<
                          OutputIteratorT>::value_type>>>>
void group_store(GroupHelper gh,
                 const sycl::span<InputT, ElementsPerWorkItem> &in,
                 OutputIteratorT out_ptr, Properties properties = {}) {
  constexpr bool blocked = detail::is_blocked(properties);
  auto generic = [&]() {
    for (int i = 0; i < in.size(); ++i)
      out_ptr[detail::get_mem_idx<blocked, ElementsPerWorkItem>(gh, i)] = in[i];
  };

  group_barrier(gh);
  generic();
  group_barrier(gh);
}
} // namespace ext::oneapi::experimental
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
