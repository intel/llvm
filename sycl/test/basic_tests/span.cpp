// RUN: %clangxx -fsycl -fsyntax-only -std=c++17 -ferror-limit=0 -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
// RUN: %clangxx -fsycl -fsyntax-only -std=c++20 -ferror-limit=0 -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

//==--------------- span.cpp - SYCL span test ------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Checks the sycl::span API: constructors, slicing, observers, element access,
// iterators, byte views, deduction guides and constraints. Valid uses are
// evaluated at compile time and also instantiated in device code. Invalid uses
// are checked with -verify.

#include <sycl/sycl.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

using sycl::dynamic_extent;
using sycl::span;

// Random access iterator that is not a pointer and not a class template, so
// __std_to_address must fall back to operator->().
struct ArrowIt {
  using iterator_category = std::random_access_iterator_tag;
  using value_type = int;
  using difference_type = std::ptrdiff_t;
  using pointer = int *;
  using reference = int &;
  int *p;
  constexpr int &operator*() const { return *p; }
  constexpr int *operator->() const { return p; }
  constexpr difference_type operator-(const ArrowIt &o) const {
    return p - o.p;
  }
};

// Random access iterator without operator->(), so __std_to_address must use
// pointer_traits<PtrTraitsIt>::to_address.
struct PtrTraitsIt {
  using iterator_category = std::random_access_iterator_tag;
  using value_type = int;
  using difference_type = std::ptrdiff_t;
  using pointer = int *;
  using reference = int &;
  int *p;
  constexpr int &operator*() const { return *p; }
  constexpr difference_type operator-(const PtrTraitsIt &o) const {
    return p - o.p;
  }
};

template <> struct std::pointer_traits<PtrTraitsIt> {
  using pointer = PtrTraitsIt;
  using element_type = int;
  using difference_type = std::ptrdiff_t;
  static constexpr int *to_address(const PtrTraitsIt &it) noexcept {
    return it.p;
  }
};

// Contiguous container that is neither an array nor a std::array, so span uses
// its container constructors. Deriving from std::array makes std::data and
// std::size visible to ADL and keeps it usable in constant expressions.
struct Container : std::array<int, 6> {};

// Checks that s views size elements starting at data.
template <class S>
constexpr void checkView(const S &s, const int *data, std::size_t size) {
  assert(s.data() == data);
  assert(s.size() == size);
}

// Each block constructs spans from one kind of source, with a static extent,
// an explicit dynamic extent, no template arguments and only the element type.
constexpr void testConstructors() {
  int a[6] = {0, 1, 2, 3, 4, 5};
  const int ca[6] = {0, 1, 2, 3, 4, 5};
  std::array<int, 6> sa = {0, 1, 2, 3, 4, 5};
  const std::array<int, 6> csa = {0, 1, 2, 3, 4, 5};
  Container c = {0, 1, 2, 3, 4, 5};
  const Container cc = {0, 1, 2, 3, 4, 5};

  // Default.
  {
    span<int, 0> staticExtent;
    span<int, dynamic_extent> dynamicExtent;
    span<int> partial;
    checkView(staticExtent, nullptr, 0);
    checkView(dynamicExtent, nullptr, 0);
    checkView(partial, nullptr, 0);
  }

  // From an array.
  {
    span<int, 6> staticExtent{a};
    span<int, dynamic_extent> dynamicExtent{a};
    span unspecialized{a};
    span<int> partial{a};
    static_assert(std::is_same_v<decltype(unspecialized), span<int, 6>>);
    static_assert(std::is_same_v<decltype(partial), span<int, dynamic_extent>>);
    checkView(staticExtent, a, 6);
    checkView(dynamicExtent, a, 6);
    checkView(unspecialized, a, 6);
    checkView(partial, a, 6);

    span<int, 6> implicitStatic = a;
    span<int> implicitDynamic = a;
    span<const int> toConst{a};
    checkView(implicitStatic, a, 6);
    checkView(implicitDynamic, a, 6);
    checkView(toConst, a, 6);
  }

  // From a const array.
  {
    span<const int, 6> staticExtent{ca};
    span<const int, dynamic_extent> dynamicExtent{ca};
    span unspecialized{ca};
    span<const int> partial{ca};
    static_assert(std::is_same_v<decltype(unspecialized), span<const int, 6>>);
    static_assert(
        std::is_same_v<decltype(partial), span<const int, dynamic_extent>>);
    checkView(staticExtent, ca, 6);
    checkView(dynamicExtent, ca, 6);
    checkView(unspecialized, ca, 6);
    checkView(partial, ca, 6);
  }

  // From a std::array.
  {
    span<int, 6> staticExtent{sa};
    span<int, dynamic_extent> dynamicExtent{sa};
    span unspecialized{sa};
    span<int> partial{sa};
    static_assert(std::is_same_v<decltype(unspecialized), span<int, 6>>);
    static_assert(std::is_same_v<decltype(partial), span<int, dynamic_extent>>);
    checkView(staticExtent, sa.data(), 6);
    checkView(dynamicExtent, sa.data(), 6);
    checkView(unspecialized, sa.data(), 6);
    checkView(partial, sa.data(), 6);

    span<int, 6> implicitStatic = sa;
    span<int> implicitDynamic = sa;
    span<const int, 6> toConst{sa};
    checkView(implicitStatic, sa.data(), 6);
    checkView(implicitDynamic, sa.data(), 6);
    checkView(toConst, sa.data(), 6);
  }

  // From a const std::array.
  {
    span<const int, 6> staticExtent{csa};
    span<const int, dynamic_extent> dynamicExtent{csa};
    span unspecialized{csa};
    span<const int> partial{csa};
    static_assert(std::is_same_v<decltype(unspecialized), span<const int, 6>>);
    static_assert(
        std::is_same_v<decltype(partial), span<const int, dynamic_extent>>);
    checkView(staticExtent, csa.data(), 6);
    checkView(dynamicExtent, csa.data(), 6);
    checkView(unspecialized, csa.data(), 6);
    checkView(partial, csa.data(), 6);
  }

  // From a container.
  {
    span<int, 6> staticExtent{c};
    span<int, dynamic_extent> dynamicExtent{c};
    span unspecialized{c};
    span<int> partial{c};
    static_assert(std::is_same_v<decltype(unspecialized), span<int>>);
    static_assert(std::is_same_v<decltype(partial), span<int, dynamic_extent>>);
    checkView(staticExtent, c.data(), 6);
    checkView(dynamicExtent, c.data(), 6);
    checkView(unspecialized, c.data(), 6);
    checkView(partial, c.data(), 6);

    span<int> implicitDynamic = c;
    span<const int> toConst{c};
    checkView(implicitDynamic, c.data(), 6);
    checkView(toConst, c.data(), 6);
  }

  // From a const container.
  {
    span<const int, 6> staticExtent{cc};
    span<const int, dynamic_extent> dynamicExtent{cc};
    span unspecialized{cc};
    span<const int> partial{cc};
    static_assert(std::is_same_v<decltype(unspecialized), span<const int>>);
    static_assert(
        std::is_same_v<decltype(partial), span<const int, dynamic_extent>>);
    checkView(staticExtent, cc.data(), 6);
    checkView(dynamicExtent, cc.data(), 6);
    checkView(unspecialized, cc.data(), 6);
    checkView(partial, cc.data(), 6);
  }

  // From a pointer and a count.
  {
    span<int, 6> staticExtent{a, 6};
    span<int, dynamic_extent> dynamicExtent{a, 4};
    span unspecialized{a, 4};
    span<int> partial{a, 4};
    static_assert(std::is_same_v<decltype(unspecialized), span<int>>);
    static_assert(std::is_same_v<decltype(partial), span<int, dynamic_extent>>);
    checkView(staticExtent, a, 6);
    checkView(dynamicExtent, a, 4);
    checkView(unspecialized, a, 4);
    checkView(partial, a, 4);

    span<int> implicitDynamic = {a, 4};
    checkView(implicitDynamic, a, 4);
  }

  // From two pointers.
  {
    span<int, 6> staticExtent{a, a + 6};
    span<int, dynamic_extent> dynamicExtent{a + 1, a + 6};
    span unspecialized{a + 1, a + 6};
    span<int> partial{a + 1, a + 6};
    static_assert(std::is_same_v<decltype(unspecialized), span<int>>);
    static_assert(std::is_same_v<decltype(partial), span<int, dynamic_extent>>);
    checkView(staticExtent, a, 6);
    checkView(dynamicExtent, a + 1, 5);
    checkView(unspecialized, a + 1, 5);
    checkView(partial, a + 1, 5);

    span<int> implicitDynamic = {a + 1, a + 6};
    checkView(implicitDynamic, a + 1, 5);
  }

  // From two iterators converted with operator->().
  {
    span<int, 6> staticExtent{ArrowIt{a}, ArrowIt{a + 6}};
    span<int, dynamic_extent> dynamicExtent{ArrowIt{a}, ArrowIt{a + 3}};
    span unspecialized{ArrowIt{a}, ArrowIt{a + 3}};
    span<int> partial{ArrowIt{a}, ArrowIt{a + 3}};
    static_assert(std::is_same_v<decltype(unspecialized), span<int>>);
    static_assert(std::is_same_v<decltype(partial), span<int, dynamic_extent>>);
    checkView(staticExtent, a, 6);
    checkView(dynamicExtent, a, 3);
    checkView(unspecialized, a, 3);
    checkView(partial, a, 3);
  }

  // From an iterator converted with pointer_traits and a count.
  {
    span<int, 6> staticExtent{PtrTraitsIt{a}, 6};
    span<int, dynamic_extent> dynamicExtent{PtrTraitsIt{a}, 5};
    span unspecialized{PtrTraitsIt{a}, 5};
    span<int> partial{PtrTraitsIt{a}, 5};
    static_assert(std::is_same_v<decltype(unspecialized), span<int>>);
    static_assert(std::is_same_v<decltype(partial), span<int, dynamic_extent>>);
    checkView(staticExtent, a, 6);
    checkView(dynamicExtent, a, 5);
    checkView(unspecialized, a, 5);
    checkView(partial, a, 5);
  }

  // From another span.
  {
    span<int, 6> fixed{a};
    span<int> dyn{a};
    span<int, 6> staticExtent{fixed};
    span<int, 6> staticFromDynamic{dyn};
    span<int, dynamic_extent> dynamicExtent = fixed;
    span unspecialized{fixed};
    span unspecializedDynamic{dyn};
    span<int> partial = fixed;
    static_assert(std::is_same_v<decltype(unspecialized), span<int, 6>>);
    static_assert(std::is_same_v<decltype(unspecializedDynamic), span<int>>);
    static_assert(std::is_same_v<decltype(partial), span<int, dynamic_extent>>);
    checkView(staticExtent, a, 6);
    checkView(staticFromDynamic, a, 6);
    checkView(dynamicExtent, a, 6);
    checkView(unspecialized, a, 6);
    checkView(unspecializedDynamic, a, 6);
    checkView(partial, a, 6);

    span<const int, 6> toConstStatic = fixed;
    span<const int> toConstDynamic = dyn;
    checkView(toConstStatic, a, 6);
    checkView(toConstDynamic, a, 6);
  }

  // Copy assignment.
  {
    span<int, 6> staticExtent{a};
    span<int> dynamicExtent{a};
    staticExtent = span<int, 6>{sa};
    dynamicExtent = span<int>{a, 4};
    checkView(staticExtent, sa.data(), 6);
    checkView(dynamicExtent, a, 4);
  }
}

// std::vector is not usable in constant expressions in C++17, so only check
// that these compile and deduce the expected types.
void testVectorConstructors() {
  std::vector<int> v(6);
  const std::vector<int> cv(6);

  // From a vector.
  {
    span<int, 6> staticExtent{v};
    span<int, dynamic_extent> dynamicExtent{v};
    span unspecialized{v};
    span<int> partial{v};
    static_assert(std::is_same_v<decltype(unspecialized), span<int>>);
    static_assert(std::is_same_v<decltype(partial), span<int, dynamic_extent>>);

    span<int> implicitDynamic = v;
    span<const int> toConst{v};
  }

  // From a const vector.
  {
    span<const int, 6> staticExtent{cv};
    span<const int, dynamic_extent> dynamicExtent{cv};
    span unspecialized{cv};
    span<const int> partial{cv};
    static_assert(std::is_same_v<decltype(unspecialized), span<const int>>);
    static_assert(
        std::is_same_v<decltype(partial), span<const int, dynamic_extent>>);
  }

  // From a temporary vector, which requires const elements.
  {
    span<const int> partial{std::vector<int>{}};
  }

  // From vector iterators.
  {
    span<int, 6> staticExtent{v.begin(), v.end()};
    span<int, dynamic_extent> dynamicExtent{v.begin(), v.end()};
    span unspecialized{v.begin(), v.end()};
    span<int> partial{v.begin(), v.end()};
    static_assert(std::is_same_v<decltype(unspecialized), span<int>>);
    static_assert(std::is_same_v<decltype(partial), span<int, dynamic_extent>>);
  }

  // From const vector iterators.
  {
    span<const int, 6> staticExtent{v.cbegin(), v.cend()};
    span<const int, dynamic_extent> dynamicExtent{v.cbegin(), v.cend()};
    span unspecialized{v.cbegin(), v.cend()};
    span<const int> partial{v.cbegin(), v.cend()};
    static_assert(std::is_same_v<decltype(unspecialized), span<const int>>);
    static_assert(
        std::is_same_v<decltype(partial), span<const int, dynamic_extent>>);
  }
}

constexpr void testStaticExtent() {
  int a[6] = {0, 1, 2, 3, 4, 5};
  span<int, 6> s{a};

  // Observers.
  assert(s.size() == 6);
  assert(s.size_bytes() == 6 * sizeof(int));
  assert(!s.empty());
  assert((span<int, 0>{}.empty()));

  // Element access.
  assert(s[2] == 2);
  assert(s.front() == 0);
  assert(s.back() == 5);
  assert(s.data() == a);
  s[2] = 20;
  assert(a[2] == 20);
  a[2] = 2;

  // Iterators.
  assert(s.end() - s.begin() == 6);
  assert(*s.begin() == 0);
  assert(s.rend() - s.rbegin() == 6);
  assert(*s.rbegin() == 5);
  int sum = 0;
  for (int e : s)
    sum += e;
  assert(sum == 15);

  // Subviews with static extents.
  auto f = s.first<2>();
  static_assert(std::is_same_v<decltype(f), span<int, 2>>);
  assert(f.data() == a);
  assert(f.size() == 2);
  auto l = s.last<2>();
  static_assert(std::is_same_v<decltype(l), span<int, 2>>);
  assert(l.data() == a + 4);
  assert(l.size() == 2);
  auto sub = s.subspan<2>();
  static_assert(std::is_same_v<decltype(sub), span<int, 4>>);
  assert(sub.data() == a + 2);
  assert(sub.size() == 4);
  auto sub2 = s.subspan<1, 3>();
  static_assert(std::is_same_v<decltype(sub2), span<int, 3>>);
  assert(sub2.data() == a + 1);
  assert(sub2.size() == 3);
  auto subEnd = s.subspan<6>();
  static_assert(std::is_same_v<decltype(subEnd), span<int, 0>>);
  assert(subEnd.empty());

  // Subviews with dynamic extents.
  auto fd = s.first(3);
  static_assert(std::is_same_v<decltype(fd), span<int>>);
  assert(fd.data() == a);
  assert(fd.size() == 3);
  auto ld = s.last(3);
  static_assert(std::is_same_v<decltype(ld), span<int>>);
  assert(ld.data() == a + 3);
  assert(ld.size() == 3);
  auto sd = s.subspan(1, 2);
  static_assert(std::is_same_v<decltype(sd), span<int>>);
  assert(sd.data() == a + 1);
  assert(sd.size() == 2);
  auto sdRest = s.subspan(4);
  assert(sdRest.data() == a + 4);
  assert(sdRest.size() == 2);
  assert(s.first(0).empty());
  assert(s.last(0).empty());
  assert(s.subspan(6).empty());
}

constexpr void testDynamicExtent() {
  int a[6] = {0, 1, 2, 3, 4, 5};
  span<int> s{a};
  span<int> def;

  // Observers.
  assert(s.size() == 6);
  assert(s.size_bytes() == 6 * sizeof(int));
  assert(!s.empty());

  // Element access.
  assert(s[2] == 2);
  assert(s.front() == 0);
  assert(s.back() == 5);
  assert(s.data() == a);
  s[2] = 20;
  assert(a[2] == 20);
  a[2] = 2;

  // Iterators.
  assert(s.end() - s.begin() == 6);
  assert(*s.begin() == 0);
  assert(s.rend() - s.rbegin() == 6);
  assert(*s.rbegin() == 5);
  assert(def.begin() == def.end());
  assert(def.rbegin() == def.rend());
  int sum = 0;
  for (int e : s)
    sum += e;
  assert(sum == 15);

  // Subviews with static extents.
  auto f = s.first<2>();
  static_assert((std::is_same_v<decltype(f), span<int, 2>>));
  assert(f.data() == a);
  assert(f.size() == 2);
  auto l = s.last<2>();
  static_assert((std::is_same_v<decltype(l), span<int, 2>>));
  assert(l.data() == a + 4);
  assert(l.size() == 2);
  auto sub = s.subspan<2>();
  static_assert((std::is_same_v<decltype(sub), span<int>>));
  assert(sub.data() == a + 2);
  assert(sub.size() == 4);
  auto sub2 = s.subspan<1, 3>();
  static_assert(std::is_same_v<decltype(sub2), span<int, 3>>);
  assert(sub2.data() == a + 1);
  assert(sub2.size() == 3);

  // Subviews with dynamic extents.
  auto fd = s.first(3);
  assert(fd.data() == a);
  assert(fd.size() == 3);
  auto ld = s.last(3);
  assert(ld.data() == a + 3);
  assert(ld.size() == 3);
  auto sd = s.subspan(1, 2);
  assert(sd.data() == a + 1);
  assert(sd.size() == 2);
  auto sdRest = s.subspan(4);
  assert(sdRest.data() == a + 4);
  assert(sdRest.size() == 2);
  assert(s.first(0).empty());
  assert(s.last(0).empty());
  assert(s.subspan(6).empty());
}

constexpr bool testAll() {
  testConstructors();
  testStaticExtent();
  testDynamicExtent();
  return true;
}

static_assert(testAll());

// Byte views are not constexpr, check their types only.
using ByteT = unsigned char;
static_assert(
    std::is_same_v<decltype(sycl::as_bytes(std::declval<span<int, 3>>())),
                   span<const ByteT, 3 * sizeof(int)>>);
static_assert(std::is_same_v<
              decltype(sycl::as_writable_bytes(std::declval<span<int, 3>>())),
              span<ByteT, 3 * sizeof(int)>>);
static_assert(
    std::is_same_v<decltype(sycl::as_bytes(span<int>{})), span<const ByteT>>);
static_assert(std::is_same_v<decltype(sycl::as_writable_bytes(span<int>{})),
                             span<ByteT>>);

// Member types and constants.
static_assert(std::is_same_v<span<const int>::element_type, const int>);
static_assert(std::is_same_v<span<const int>::value_type, int>);
static_assert(std::is_same_v<span<int>::size_type, std::size_t>);
static_assert(std::is_same_v<span<int>::difference_type, std::ptrdiff_t>);
static_assert(std::is_same_v<span<int>::pointer, int *>);
static_assert(std::is_same_v<span<int>::const_pointer, const int *>);
static_assert(std::is_same_v<span<int>::reference, int &>);
static_assert(std::is_same_v<span<int>::const_reference, const int &>);
static_assert(std::is_same_v<span<int>::reverse_iterator,
                             std::reverse_iterator<span<int>::iterator>>);
static_assert(span<int>::extent == dynamic_extent);
static_assert(span<int, 4>::extent == 4);
static_assert(std::is_trivially_copyable_v<span<int>>);
static_assert(std::is_trivially_copyable_v<span<int, 4>>);
static_assert(std::is_nothrow_default_constructible_v<span<int>>);
static_assert(std::is_nothrow_default_constructible_v<span<int, 0>>);

void testInvalid() {
  int arr[4]{};
  const int constArr[4]{};
  long longArr[4]{};
  std::array<int, 4> stdArr{};
  const std::array<int, 4> constStdArr{};
  std::vector<int> vec(4);
  const std::vector<int> constVec(4);
  span<int> dyn{arr};
  span<const int> constDyn{arr};
  span<int, 4> fixed{arr};
  span<int, 3> fixed3{arr, 3};

  // Default construction is only available for zero or dynamic extents.
  // expected-error@+1 {{no matching constructor}}
  span<int, 4> noDefault;

  // Construction from arrays requires matching extents and compatible
  // elements.
  // expected-error@+1 {{no matching constructor}}
  span<int, 3> arrTooBig{arr};
  // expected-error@+1 {{no matching constructor}}
  span<int, 5> arrTooSmall{arr};
  // expected-error@+1 {{no matching constructor}}
  span<int> arrDropsConst{constArr};
  // expected-error@+1 {{no matching constructor}}
  span<long> arrWrongType{arr};
  // expected-error@+1 {{no matching constructor}}
  span<int, 4> stdArrDropsConst{constStdArr};
  // expected-error@+1 {{no matching constructor}}
  span<int> dynStdArrDropsConst{constStdArr};
  // expected-error@+1 {{no matching constructor}}
  span<int, 3> stdArrWrongSize{stdArr};

  // Construction from containers requires compatible elements.
  // expected-error@+1 {{no matching constructor}}
  span<int> vecDropsConst{constVec};
  // expected-error@+1 {{no matching constructor}}
  span<long> vecWrongType{vec};
  // expected-error@+1 {{no matching constructor}}
  span<int> fromRvalueVec{std::vector<int>{}};

  // Construction from iterators requires compatible elements.
  // expected-error@+1 {{no matching constructor}}
  span<int> ptrCountDropsConst{constArr, 4};
  // expected-error@+1 {{no matching constructor}}
  span<int> ptrPtrDropsConst{constArr, constArr + 4};
  // expected-error@+1 {{no matching constructor}}
  span<int> ptrPtrWrongType{longArr, longArr + 4};
  // expected-error@+1 {{no matching constructor}}
  span<int> constIterDropsConst{vec.cbegin(), vec.cend()};

  // Conversions between spans require compatible elements and extents.
  // expected-error@+1 {{no matching constructor}}
  span<int> spanDropsConst{constDyn};
  // expected-error@+1 {{no matching constructor}}
  span<long> spanWrongType{dyn};
  // expected-error@+1 {{no matching constructor}}
  span<int, 4> spanWrongExtent{fixed3};

  // These constructors are explicit for static extents.
  // expected-error@+1 {{chosen constructor is explicit in copy-initialization}}
  span<int, 4> implicitPtrCount = {arr, 4};
  // expected-error@+1 {{chosen constructor is explicit in copy-initialization}}
  span<int, 4> implicitPtrPtr = {arr, arr + 4};
  // expected-error@+1 {{no viable conversion}}
  span<int, 4> implicitFromVec = vec;
  // expected-error@+1 {{no viable conversion}}
  span<int, 4> implicitFromDynamic = dyn;

  // Writable byte views require non-const elements.
  // expected-error@+1 {{no matching function for call to 'as_writable_bytes'}}
  sycl::as_writable_bytes(constDyn);
  // expected-error@+1 {{no matching function for call to 'as_writable_bytes'}}
  sycl::as_writable_bytes(span<const int, 4>{fixed});

  // Static subviews of static extents are bounds-checked at compile time.
  // expected-error@sycl/sycl_span.hpp:* {{Count out of range in span::first()}}
  (void)fixed.first<5>();
  // expected-error@sycl/sycl_span.hpp:* {{Count out of range in span::last()}}
  (void)fixed.last<5>();
  // expected-error@sycl/sycl_span.hpp:* {{Offset out of range in span::subspan()}}
  (void)fixed.subspan<5>();
  // expected-error@sycl/sycl_span.hpp:* {{Offset + count out of range in span::subspan()}}
  (void)fixed.subspan<1, 4>();
}

// Precondition checks make evaluation non-constant. They are disabled in device
// code.
#ifndef __SYCL_DEVICE_ONLY__
constexpr int constArr[4]{1, 2, 3, 4};
constexpr span<const int> constSpan{constArr};
constexpr span<const int> emptySpan{};
// expected-error@+1 {{must be initialized by a constant expression}}
constexpr span<const int, 4> badPtrCount{constArr, 3};
// expected-error@+1 {{must be initialized by a constant expression}}
constexpr span<const int, 4> badPtrPtr{constArr, constArr + 3};
// expected-error@+1 {{must be initialized by a constant expression}}
constexpr span<const int, 3> badFromDynamic{constSpan};
// expected-error@+1 {{must be initialized by a constant expression}}
constexpr auto badFirst = constSpan.first(5);
// expected-error@+1 {{must be initialized by a constant expression}}
constexpr auto badLast = constSpan.last(5);
// expected-error@+1 {{must be initialized by a constant expression}}
constexpr auto badSubspan = constSpan.subspan(5);
// expected-error@+1 {{must be initialized by a constant expression}}
constexpr auto badSubspanCount = constSpan.subspan(2, 3);
// expected-error@+1 {{must be initialized by a constant expression}}
constexpr auto badFirstStatic = constSpan.first<5>();
// expected-error@+1 {{must be initialized by a constant expression}}
constexpr auto badLastStatic = constSpan.last<5>();
// expected-error@+1 {{must be initialized by a constant expression}}
constexpr auto badSubspanStatic = constSpan.subspan<5>();
// expected-error@+1 {{must be initialized by a constant expression}}
constexpr int badIndex = constSpan[4];
// expected-error@+1 {{must be initialized by a constant expression}}
constexpr int badFront = emptySpan.front();
// expected-error@+1 {{must be initialized by a constant expression}}
constexpr int badBack = emptySpan.back();
#endif

int main() {
  sycl::queue q;
  q.single_task([] {
    (void)testAll();
    int a[4] = {1, 2, 3, 4};
    span<int, 4> s{a};
    auto b = sycl::as_bytes(s);
    auto wb = sycl::as_writable_bytes(span<int>{s});
    (void)b;
    (void)wb;
  });
  return 0;
}
