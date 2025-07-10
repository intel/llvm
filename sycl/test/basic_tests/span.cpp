// RUN: %clangxx -fsycl %s -o %t.out && %t.out

//==--------------- span.cpp - SYCL span test ------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <sycl/sycl.hpp>
#include <sycl/sycl_span.hpp>
#include <tuple>
#include <type_traits>

template <typename T> struct requires_dynamic_memory : std::false_type {};

template <typename T, typename Alloc>
struct requires_dynamic_memory<std::vector<T, Alloc>> : std::true_type {};

template <typename T>
inline constexpr bool requires_dynamic_memory_v =
    requires_dynamic_memory<std::decay_t<T>>::value;

template <typename T> void verify_extent_deduction(const T &container) {
  sycl::span cdat{container};
  const auto size{cdat.size()};

  for (size_t i{}; i != size; i++)
    assert(cdat[i] == container[i]);

  if constexpr (not requires_dynamic_memory_v<T>) {
    if constexpr (std::is_array_v<T>)
      static_assert(decltype(cdat)::extent == std::extent_v<T>,
                    "Extent should match c-array size");
    else
      static_assert(decltype(cdat)::extent == std::tuple_size_v<T>,
                    "Extent should match std::array size");
  } else {
    assert(cdat.size() == container.size());
    static_assert(
        decltype(cdat)::extent == sycl::dynamic_extent,
        "Extent should be dynamic for containers with dynamic memory");
  }
}

int main() {
  int arr[]{1, 2, 3, 4};
  const int constArr[]{8, 7, 6};
  std::vector<int> vec(4);

  // various span declarations, especially unspecialized
  sycl::span fromArray{arr};
  sycl::span fromConstArray{constArr};
  sycl::span fromVec{vec};

  // partly specialized
  sycl::span<int> fromIntArray{arr};
  sycl::span<const int> fromIntConstArray{constArr};
  sycl::span<int> fromIntVec{vec};

  // fully specialized
  sycl::span<int, 4> fullSpecArray{arr};
  sycl::span<const int, 3> fullSpecConstArray{constArr};
  sycl::span<int, 4> fullSpecVecArray{vec};

  // check that the extent is deduced correctly
  static_assert(decltype(fromArray)::extent == decltype(fullSpecArray)::extent,
                "extent doesn't match between unspecialized and fully "
                "specialized span from array");
  static_assert(decltype(fromConstArray)::extent ==
                    decltype(fullSpecConstArray)::extent,
                "extent doesn't match between unspecialized"
                "and fully specialized span from const array");
  static_assert(
      std::is_same_v<
          std::integral_constant<size_t, decltype(fromConstArray)::extent>,
          std::integral_constant<size_t, decltype(fullSpecConstArray)::extent>>,
      "extent doesn't match between unspecialized"
      "and fully specialized span from const array");

  static_assert(std::is_default_constructible_v<sycl::span<int>>);
  static_assert(std::is_default_constructible_v<sycl::span<int, 0>>);
  static_assert(!std::is_default_constructible_v<sycl::span<int, 5>>);

  /*
  C++20 span explicitness rule:
    A constructor is explicit if both of the following are true:
      1. It's a converting constructor (not from array/std::array)
      2. The span has fixed extent which != dynamic_extent

  Test explanation:
  - is_constructible: can construct using explicit constructor syntax Type(args)
  - is_convertible: can construct using implicit conversion Type var = args*/
  {
    // ALWAYS IMPLICIT (dynamic, fixed extent)
    { // array reference constructors
      static_assert(std::is_constructible_v<sycl::span<int>, int (&)[4]>);
      static_assert(std::is_convertible_v<int (&)[4], sycl::span<int>>);
      static_assert(std::is_constructible_v<sycl::span<int, 4>, int (&)[4]>);
      static_assert(std::is_convertible_v<int (&)[4], sycl::span<int, 4>>);
    }
    { // std::array constructors
      static_assert(
          std::is_constructible_v<sycl::span<int>, std::array<int, 4> &>);
      static_assert(
          std::is_convertible_v<std::array<int, 4> &, sycl::span<int>>);
      static_assert(
          std::is_constructible_v<sycl::span<int, 4>, std::array<int, 4> &>);
      static_assert(
          std::is_convertible_v<std::array<int, 4> &, sycl::span<int, 4>>);
    }
    { // pointer + count constructors: explicit only when fixed extent
      static_assert(std::is_constructible_v<sycl::span<int>, int *, size_t>);
      static_assert(std::is_constructible_v<sycl::span<int, 4>, int *, size_t>);
    }
    { // container constructor: explicit only when fixed extent
      static_assert(
          std::is_constructible_v<sycl::span<int>, std::vector<int> &>);
      static_assert(std::is_convertible_v<std::vector<int> &, sycl::span<int>>);
      static_assert(
          std::is_constructible_v<sycl::span<int, 4>, std::vector<int> &>);
      static_assert(
          !std::is_convertible_v<std::vector<int> &, sycl::span<int, 4>>);
    }
    { // copy constructor: always implicit
      static_assert(std::is_constructible_v<sycl::span<int>, sycl::span<int>>);
      static_assert(std::is_convertible_v<sycl::span<int>, sycl::span<int>>);
      static_assert(
          std::is_constructible_v<sycl::span<int, 4>, sycl::span<int, 4>>);
      static_assert(
          std::is_convertible_v<sycl::span<int, 4>, sycl::span<int, 4>>);
    }
    { // converting constructor between spans: fixed->dynamic (implicit)
      static_assert(
          std::is_constructible_v<sycl::span<int>, sycl::span<int, 4>>);
      static_assert(std::is_convertible_v<sycl::span<int, 4>, sycl::span<int>>);

      // Const conversions
      static_assert(
          std::is_constructible_v<sycl::span<const int>, sycl::span<int>>);
      static_assert(std::is_constructible_v<sycl::span<const int, 4>,
                                            sycl::span<int, 4>>);
    }
    { // initializer_list constructor: implicit for dynamic (deviates from
      // C++20), explicit for fixed extent
      // Note: C++20 spec requires explicit for ALL extents, but SYCL keeps
      // dynamic extent implicit
      static_assert(std::is_constructible_v<sycl::span<const int>,
                                            std::initializer_list<int>>);
      static_assert(
          std::is_convertible_v<std::initializer_list<int>,
                                sycl::span<const int>>); // implicit for dynamic
      static_assert(std::is_constructible_v<sycl::span<const int, 3>,
                                            std::initializer_list<int>>);
      static_assert(
          !std::is_convertible_v<std::initializer_list<int>,
                                 sycl::span<const int, 3>>); // explicit for
                                                             // fixed
    }
  }

  { // correctness of iterator types types and traits
    sycl::span<int> sp(arr, 4);
    static_assert(std::is_same_v<decltype(sp.begin()),
                                 typename sycl::span<int>::iterator>);
    static_assert(
        std::is_same_v<decltype(sp.end()), typename sycl::span<int>::iterator>);
    static_assert(std::is_same_v<decltype(sp.cbegin()),
                                 typename sycl::span<int>::const_iterator>);
    static_assert(std::is_same_v<decltype(sp.cend()),
                                 typename sycl::span<int>::const_iterator>);
    static_assert(std::is_same_v<decltype(sp.rbegin()),
                                 typename sycl::span<int>::reverse_iterator>);
    static_assert(std::is_same_v<decltype(sp.rend()),
                                 typename sycl::span<int>::reverse_iterator>);
    static_assert(
        std::is_same_v<decltype(sp.crbegin()),
                       typename sycl::span<int>::const_reverse_iterator>);
    static_assert(
        std::is_same_v<decltype(sp.crend()),
                       typename sycl::span<int>::const_reverse_iterator>);

    using iter_traits = std::iterator_traits<decltype(sp.begin())>;
    static_assert(std::is_same_v<iter_traits::iterator_category,
                                 std::random_access_iterator_tag>);
    static_assert(std::is_same_v<iter_traits::value_type, int>);
    static_assert(std::is_same_v<iter_traits::pointer, int *>);
    static_assert(std::is_same_v<iter_traits::reference, int &>);
  }

  { // test fixed vs dynamic extent
    using fixed_span_t = sycl::span<int, 4>;
    using dynamic_span_t = sycl::span<int>;
    static_assert(fixed_span_t::extent == 4);
    static_assert(dynamic_span_t::extent == sycl::dynamic_extent);

    using fixed_const_span_t = sycl::span<const int, 4>;
    using dynamic_const_span_t = sycl::span<const int>;
    static_assert(fixed_const_span_t::extent == 4);
    static_assert(dynamic_const_span_t::extent == sycl::dynamic_extent);
  }

  { // test const correctness
    const int const_arr[3] = {1, 2, 3};
    sycl::span<const int> const_span(const_arr);
    static_assert(
        std::is_same_v<decltype(const_span)::element_type, const int>);
  }

  { // test new C++20 features
    // Test iterator constructors
    std::vector<int> vec2 = {1, 2, 3, 4, 5};
    sycl::span<int> sp1(vec2.begin(), size_t(3));
    sycl::span<int> sp2(vec2.begin(), vec2.end());

    // Test fixed extent iterator constructor
    sycl::span<int, 3> sp3(vec2.begin(), size_t(3));

    // Test initializer_list constructor
    std::initializer_list<int> il = {10, 20, 30};
    sycl::span<const int> sp4(il);
    sycl::span<const int, 3> sp5(il);

    // Test that at() method exists
    int arr2[3] = {1, 2, 3};
    sycl::span<int> sp(arr2);
    static_assert(std::is_same_v<decltype(sp.at(0)), int &>);

    // Test iterator deduction guides (C++17 and later)
#if __cplusplus >= 201703L
    sycl::span sp6{vec2.begin(), vec2.end()};
    static_assert(std::is_same_v<decltype(sp6), sycl::span<int>>);

    sycl::span sp7{vec2.begin(), size_t(3)};
    static_assert(std::is_same_v<decltype(sp7), sycl::span<int>>);

    // Test range deduction guide
    sycl::span sp8{vec2};
    static_assert(std::is_same_v<decltype(sp8), sycl::span<int>>);
#endif
  }

  verify_extent_deduction((int[5]){1, 2, 3, 4, 5});
  verify_extent_deduction(std::vector{10, 200, 30, 400});
  verify_extent_deduction(std::array{100, 20, 300, 40});

  return 0;
}
