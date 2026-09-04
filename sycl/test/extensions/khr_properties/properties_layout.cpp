// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics
//
// Checks the size and trivial-copyability guarantees of khr::properties. The
// extension does not mandate these, but we ensure: compile-time-only lists are
// minimal size, mixed lists pay only for their runtime members, and lists are
// trivially copyable (unlike a std::tuple-based implementation).

#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS
#include <sycl/khr/properties.hpp>

namespace kd = sycl::khr::detail;
using namespace sycl::khr;

struct rt1_key : kd::runtime_property_key {};
struct rt1 : kd::runtime_property<rt1_key> {
  bool value;
  constexpr rt1(bool v = true) : value(v) {}
};
struct rt2_key : kd::runtime_property_key {};
struct rt2 : kd::runtime_property<rt2_key> {
  bool value;
  constexpr rt2(bool v = true) : value(v) {}
};

struct ct1_key : kd::constant_value_property_key {};
template <int A>
inline constexpr ct1_key::__detail_property_t<ct1_key, int, A> ct1;
struct ct2_key : kd::constant_value_property_key {};
template <int A>
inline constexpr ct2_key::__detail_property_t<ct2_key, int, A> ct2;

struct hy_key : kd::hybrid_property_key {};
template <int X> struct hy : kd::hybrid_property<hy_key> {
  static constexpr int x = X;
  int y;
  constexpr hy(int y) : y(y) {}
};

// Individual properties.
static_assert(sizeof(rt1{true}) == sizeof(bool));
static_assert(sizeof(ct1<16>) == 1);
static_assert(sizeof(hy<1>{2}) == sizeof(int));

// Lists: compile-time-only are minimal; runtime/hybrid pay only for their
// runtime members; empty-base duplication does not bloat the list.
static_assert(sizeof(properties{}) == 1);
static_assert(sizeof(properties{ct1<16>, ct2<8>}) == 1);
static_assert(sizeof(properties{rt1{true}, rt2{false}}) == 2 * sizeof(bool));
static_assert(sizeof(properties{rt1{true}, ct1<16>, ct2<8>}) == sizeof(bool));
static_assert(sizeof(properties{hy<1>{2}, ct1<16>}) == sizeof(int));

// Trivial copyability (a std::tuple-based list would fail these).
static_assert(std::is_trivially_copyable_v<rt1>);
static_assert(std::is_trivially_copyable_v<hy<1>>);
static_assert(std::is_trivially_copyable_v<decltype(properties{})>);
static_assert(
    std::is_trivially_copyable_v<decltype(properties{rt1{true}, ct1<16>})>);
static_assert(std::is_trivially_copyable_v<decltype(properties{
                  hy<1>{2}, rt1{true}, ct1<16>})>);
