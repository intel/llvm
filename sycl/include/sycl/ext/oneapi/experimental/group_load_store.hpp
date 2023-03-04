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

  // TODO: HIP?
  constexpr bool intel_block_read_supported =
      detail::is_spir && std::is_same_v<Group, sub_group>;
  using value_type = std::iterator_traits<InputIteratorT>::value_type;
  using input_iter_no_cv = std::remove_cv_t<InputIteratorT>;

  constexpr auto size = sizeof(value_type);
  constexpr bool supported_size =
      size == 1 || size == 2 || size == 4 || size == 8;

  if constexpr (!intel_block_read_supported || !supported_size) {
    return generic();
  } else if constexpr (detail::is_multi_ptr_v<InputIteratorT>) {
    group_load(g, in_ptr.get_decorated(), out, properties);
  } else if constexpr (!std::is_pointer_v<input_iter_no_cv>) {
    return generic();
  } else {
    // Pointer.
    constexpr auto AS = sycl::detail::deduce_AS<input_iter_no_cv>::value;
    if constexpr (AS == access::address_space::global_space) {
      using BlockT = sycl::detail::sub_group::SelectBlockT<value_type>;
      using PtrT = sycl::detail::DecoratedType<BlockT, AS>::type *;

      BlockT load = __spirv_SubgroupBlockReadINTEL<BlockT>(
          reinterpret_cast<PtrT>(in_ptr));
      out = sycl::bit_cast<value_type>(load);
      return;

    } else if constexpr (AS == access::address_space::generic_space) {
      if (auto global_ptr = __SYCL_GenericCastToPtrExplicit_ToGlobal<
              remove_decoration_t<value_type>>(in_ptr))
        return group_load(g, global_ptr, out, properties);

      return generic();
    } else {
      return generic();
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
  // TODO: Consider delegating to sycl::span version via sycl::bit_cast.
  constexpr bool blocked = detail::is_blocked(properties);
  auto generic = [&]() {
    sycl::detail::dim_loop<N>([&](size_t i) {
      out[i] = in_ptr[detail::get_mem_idx<blocked, N>(g, i)];
    });
  };

  return generic();
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
