// Implements https://github.com/intel/llvm/pull/7593

#pragma once

#include <sycl/ext/oneapi/properties/properties.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext::oneapi::experimental {
namespace detail {
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
  // Default implementation.
  auto generic = [&]() { out = in_ptr[g.get_local_linear_id()]; };

  using value_type = std::iterator_traits<InputIteratorT>::value_type;
  using input_iter_no_cv = std::remove_cv_t<InputIteratorT>;

  constexpr auto size = sizeof(value_type);
  constexpr bool supported_size =
      size == 1 || size == 2 || size == 4 || size == 8;

  bool is_aligned = [&]() {
    constexpr int reqd_read_align = 4; // bytes.
    if constexpr (alignof(value_type) >= reqd_read_align)
      return true;
    else
      return (reinterpret_cast<uintptr_t>(in_ptr) % reqd_read_align) != 0;
  }();

  // TODO: HIP?
  if constexpr (!detail::is_spir || !supported_size) {
    return generic();
  } else if constexpr (detail::is_multi_ptr_v<InputIteratorT>) {
    return group_load(g, in_ptr.get_decorated(), out, properties);
  } else if constexpr (!std::is_pointer_v<input_iter_no_cv>) {
    return generic();
  } else {
    // Pointer.
    if constexpr (std::is_same_v<Group, sub_group>) {
      constexpr auto AS = sycl::detail::deduce_AS<input_iter_no_cv>::value;
      if constexpr (AS == access::address_space::global_space) {
        if (!is_aligned)
          // Not properly aligned.
          return generic();

        using BlockT = sycl::detail::sub_group::SelectBlockT<value_type>;
        using PtrT = sycl::detail::DecoratedType<BlockT, AS>::type *;

        BlockT load = __spirv_SubgroupBlockReadINTEL<BlockT>(
            reinterpret_cast<PtrT>(in_ptr));
        out = sycl::bit_cast<value_type>(load);
        return;

      } else if constexpr (AS == access::address_space::generic_space) {
        if (!is_aligned)
          // Not properly aligned.
          return generic();

        if (auto global_ptr = __SYCL_GenericCastToPtrExplicit_ToGlobal<
                remove_decoration_t<value_type>>(in_ptr))
          return group_load(g, global_ptr, out, properties);

        return generic();
      } else {
        return generic();
      }
    } else {
      // TODO: Use get_child_group from sycl_ext_oneapi_root_group extension
      // once it is implemented instead of this free function.
      auto ndi =
          sycl::ext::oneapi::experimental::this_nd_item<Group::dimensions>();
      auto sg = ndi.get_sub_group();
      // TODO: Do we have guarantees that all SGs are of the same size? Should
      // get_max_local_range be used instead?
      return group_load(sg, in_ptr + sg.get_group_id() * sg.get_local_range(),
                        out, properties);
    }
  }
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
                 Properties = {}) {
#ifdef __SYCL_DEVICE_ONLY__
  auto generic = [&]() { out_ptr[g.get_local_linear_id()] = in; };
  return generic();
#else
  throw sycl::exception(
      std::error_code(PI_ERROR_INVALID_DEVICE, sycl::sycl_category()),
      "Group loads/stoes are not supported on host.");
#endif
}

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
} // namespace detail

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
  // TODO: Consider delegating to sycl::span version via sycl::bit_cast.
  constexpr bool blocked = detail::is_blocked(properties);
  auto generic = [&]() {
    sycl::detail::dim_loop<N>([&](size_t i) {
      out[i] = in_ptr[detail::get_mem_idx<blocked, N>(g, i)];
    });
  };

  using value_type = std::iterator_traits<InputIteratorT>::value_type;
  using input_iter_no_cv = std::remove_cv_t<InputIteratorT>;

  constexpr auto size = sizeof(value_type);
  constexpr bool supported_size =
      size == 1 || size == 2 || size == 4 || size == 8;
  constexpr bool unsupported_vec_size = (N == 3 || (N == 16 && size != 1));

  bool is_aligned = [&]() {
    // TODO: Can we assume that the data is aligned as `alignof(decltype(out))`?
    constexpr int reqd_read_align = 4; // bytes.
    if constexpr (alignof(value_type) >= reqd_read_align)
      return true;
    else
      return (reinterpret_cast<uintptr_t>(in_ptr) % reqd_read_align) != 0;
  }();

  if constexpr (!detail::is_spir || !supported_size || unsupported_vec_size) {
    // TODO: Implement "unsupported_vec_size" by multiple loads followed by
    // shuffles?
    return generic();
  } else if constexpr (detail::is_multi_ptr_v<InputIteratorT>) {
    return group_load(g, in_ptr.get_decorated(), out, properties);
  } else if constexpr (!std::is_pointer_v<input_iter_no_cv>) {
    return generic();
  } else {
    // Pointer.
    if constexpr (std::is_same_v<Group, sub_group>) {
      constexpr auto AS = sycl::detail::deduce_AS<input_iter_no_cv>::value;
      if constexpr (AS == access::address_space::global_space) {
        if (!is_aligned)
          // Not properly aligned.
          return generic();

        using BlockT = sycl::detail::sub_group::SelectBlockT<value_type>;
        using VecT = sycl::detail::ConvertToOpenCLType_t<vec<BlockT, N>>;
        using PtrT = sycl::detail::DecoratedType<BlockT, AS>::type *;

        VecT load = __spirv_SubgroupBlockReadINTEL<VecT>(
            reinterpret_cast<PtrT>(in_ptr));
        auto tmp = sycl::bit_cast<vec<remove_decoration_t<value_type>, N>>(load);
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
          // clang-format on
          vec<OutputT, N> shuffled = {0, 0};
          auto lid = g.get_local_id();
          // TODO: max?
          int sg_size = g.get_max_local_range().size();
          sycl::detail::dim_loop<N>([&](size_t i) {
            shuffled[i] = select_from_group(g, tmp[(lid * N + i) / sg_size],
                                            (lid * N + i) % sg_size);
            // shuffled[i + 1] =
            //     (tmp[(lid * N + i) / sg_size]) * 1000 + (lid * N + i) % sg_size;
          });
          out = shuffled;
        } else {
          out = tmp;
        }
        return;
      } else if constexpr (AS == access::address_space::generic_space) {
        if (!is_aligned)
          // Not properly aligned.
          return generic();

        if (auto global_ptr = __SYCL_GenericCastToPtrExplicit_ToGlobal<
                remove_decoration_t<value_type>>(in_ptr))
          return group_load(g, global_ptr, out, properties);

        return generic();
      } else {
        return generic();
      }
    } else {
        return generic();
      // TODO: Use get_child_group from sycl_ext_oneapi_root_group extension
      // once it is implemented instead of this free function.
      auto ndi =
          sycl::ext::oneapi::experimental::this_nd_item<Group::dimensions>();
      auto sg = ndi.get_sub_group();
      // TODO: Do we have guarantees that all SGs are of the same size? Should
      // get_max_local_range be used instead?
      return group_load(sg,
                        in_ptr + sg.get_group_id() * sg.get_local_range() * N,
                        out, properties);
    }
  }
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

  return generic();
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
    sycl::detail::dim_loop<ElementsPerWorkItem>([&](size_t i) {
      out[i] = in_ptr[detail::get_mem_idx<blocked, ElementsPerWorkItem>(gh, i)];
    });
  };

  return generic();
}

// Store API sycl::span overload
template <typename GroupHelper, typename InputT, int ElementsPerWorkItem,
          typename OutputIteratorT,
          typename Properties = decltype(properties()),
          typename = std::enable_if_t<std::is_convertible_v<
              InputT, remove_decoration_t<typename std::iterator_traits<
                          OutputIteratorT>::value_type>>>>
void group_store(GroupHelper gh,
                 const sycl::span<InputT, ElementsPerWorkItem> &in,
                 OutputIteratorT out_ptr, Properties properties = {}) {
  constexpr bool blocked = detail::is_blocked(properties);
  auto generic = [&]() {
    sycl::detail::dim_loop<ElementsPerWorkItem>([&](size_t i) {
      out_ptr[detail::get_mem_idx<blocked, ElementsPerWorkItem>(gh, i)] = in[i];
    });
  };

  return generic();
}
} // namespace ext::oneapi::experimental
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
