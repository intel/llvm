//==----- group_interface.hpp --- sycl_khr_group_interface extension -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/id.hpp>
#include <sycl/range.hpp>

#if __cplusplus >= 202302L && defined(__has_include)
#if __has_include(<mdspan>)
#include <mdspan>
#endif
#endif

namespace sycl {
inline namespace _V1 {

namespace khr {

// Forward declarations for traits.
template <int Dimensions> class work_group;
class sub_group;
template <typename ParentGroup> class work_item;

} // namespace khr

namespace detail {
#if defined(__cpp_lib_mdspan)
template <typename IndexType, int Dimensions> struct single_extents;

template <typename IndexType> single_extents<1> {
  using type = std::extents<IndexType, 1>;
}

template <typename IndexType> single_extents<2> {
  using type = std::extents<IndexType, 1, 1>;
}

template <typename IndexType> single_extents<3> {
  using type = std::extents<IndexType, 1, 1, 1>;
}
#endif

template <typename T> struct is_khr_group : public std::false_type {};

template <int Dimensions>
struct is_khr_group<khr::work_group<Dimensions>> : public std::true_type {};

template <> struct is_khr_group<khr::sub_group> : public std::true_type {};

} // namespace detail

namespace khr {

// Forward declaration for friend function.
template <typename ParentGroup>
std::enable_if_t<detail::is_khr_group<ParentGroup>::value,
                 work_item<ParentGroup>>
get_item(ParentGroup g);

template <int Dimensions = 1> class work_group {
public:
  using id_type = id<Dimensions>;
  using linear_id_type = size_t;
  using range_type = range<Dimensions>;
#if defined(__cpp_lib_mdspan)
  using extents_type = std::dextents<size_t, Dimensions>;
#endif
  using size_type = size_t;
  static constexpr int dimensions = Dimensions;
  static constexpr memory_scope fence_scope = memory_scope::work_group;

  work_group(group<Dimensions> g) noexcept {}

  operator group<Dimensions>() const noexcept { return legacy(); }

  id_type id() const noexcept { return legacy().get_group_id(); }

  linear_id_type linear_id() const noexcept {
    return legacy().get_group_linear_id();
  }

  range_type range() const noexcept { return legacy().get_group_range(); }

#if defined(__cpp_lib_mdspan)
  constexpr extents_type extents() const noexcept {
    auto LocalRange = legacy().get_local_range();
    if constexpr (dimensions == 1) {
      return extents_type(LocalRange[0]);
    } else if constexpr (dimensions == 2) {
      return extents_type(LocalRange[0], LocalRange[1]);
    } else if constexpr (dimensions == 3) {
      return extents_type(LocalRange[0], LocalRange[1], LocalRange[2]);
    }
  }

  constexpr index_type extent(rank_type r) const noexcept {
    return extents().extent(r);
  }
#endif

  constexpr size_type size() const noexcept {
    return legacy().get_local_range().size();
  }

private:
  group<Dimensions> legacy() const noexcept {
    return ext::oneapi::this_work_item::get_work_group<Dimensions>();
  }
};

class sub_group {
public:
  using id_type = id<1>;
  using linear_id_type = uint32_t;
  using range_type = range<1>;
#if defined(__cpp_lib_mdspan)
  using extents_type = std::dextents<uint32_t, 1>;
#endif
  using size_type = uint32_t;
  static constexpr int dimensions = 1;
  static constexpr memory_scope fence_scope = memory_scope::sub_group;

  sub_group(sycl::sub_group g) noexcept {}

  operator sycl::sub_group() const noexcept { return legacy(); }

  id_type id() const noexcept { return legacy().get_group_id(); }

  linear_id_type linear_id() const noexcept {
    return legacy().get_group_linear_id();
  }

  range_type range() const noexcept { return legacy().get_group_range(); }

#if defined(__cpp_lib_mdspan)
  constexpr extents_type extents() const noexcept {
    return extents_type(legacy().get_local_range()[0]);
  }

  constexpr index_type extent(rank_type r) const noexcept {
    return extents().extent(r);
  }
#endif

  constexpr size_type size() const noexcept {
    return legacy().get_local_range()[0];
  }

  constexpr size_type max_size() const noexcept {
    return legacy().get_max_local_range()[0];
  }

private:
  sycl::sub_group legacy() const noexcept {
    return ext::oneapi::this_work_item::get_sub_group();
  }
};

template <typename ParentGroup> class work_item {
public:
  using id_type = typename ParentGroup::id_type;
  using linear_id_type = typename ParentGroup::linear_id_type;
  using range_type = typename ParentGroup::range_type;
#if defined(__cpp_lib_mdspan)
  using extents_type =
      detail::single_extents<typename ParentGroup::extents_type::index_type,
                             ParentGroup::dimensions>;
#endif
  using size_type = typename ParentGroup::size_type;
  static constexpr int dimensions = ParentGroup::dimensions;
  static constexpr memory_scope fence_scope = memory_scope::work_item;

  id_type id() const noexcept { return legacy().get_local_id(); }

  linear_id_type linear_id() const noexcept {
    return legacy().get_local_linear_id();
  }

  range_type range() const noexcept { return legacy().get_local_range(); }

#if defined(__cpp_lib_mdspan)
  constexpr extents_type extents() const noexcept { return extents_type(); }

  constexpr index_type extent(rank_type r) const noexcept {
    return extents().extent(r);
  }
#endif

  constexpr size_type size() const noexcept { return 1; }

private:
  auto legacy() const noexcept {
    if constexpr (std::is_same_v<ParentGroup, sub_group>) {
      return ext::oneapi::this_work_item::get_sub_group();
    } else {
      return ext::oneapi::this_work_item::get_work_group<
          ParentGroup::dimensions>();
    }
  }

protected:
  work_item() {}

  friend work_item<ParentGroup> get_item<ParentGroup>(ParentGroup);
};

template <typename ParentGroup>
std::enable_if_t<detail::is_khr_group<ParentGroup>::value,
                 work_item<ParentGroup>>
get_item(ParentGroup g) {
  return work_item<ParentGroup>{};
}

template <typename Group> bool leader_of(Group g) {
  return get_item(g).linear_id() == 0;
}

} // namespace khr
} // namespace _V1
} // namespace sycl
