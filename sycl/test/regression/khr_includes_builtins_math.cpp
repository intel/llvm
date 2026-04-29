// RUN: %clangxx -fsycl -fsyntax-only -Wno-deprecated-declarations %s

#include <sycl/khr/includes/builtins_math.hpp>
#include <sycl/marray.hpp>
#include <sycl/multi_ptr.hpp>
#include <sycl/vector.hpp>

#include <type_traits>

using Vec2f = sycl::vec<float, 2>;
using MArray2f = sycl::marray<float, 2>;

// Verify the KHR math wrapper exposes scalar, half, vec, and marray overloads.
static_assert(std::is_same_v<decltype(sycl::fmin(1.0f, 2.0f)), float>);
static_assert(
    std::is_same_v<decltype(sycl::sqrt(sycl::half{4.0f})), sycl::half>);
static_assert(
    std::is_same_v<decltype(sycl::fmin(Vec2f{1.0f, 2.0f}, Vec2f{2.0f, 1.0f})),
                   Vec2f>);
static_assert(std::is_same_v<decltype(sycl::fmin(MArray2f{1.0f, 2.0f},
                                                 MArray2f{2.0f, 1.0f})),
                             MArray2f>);
static_assert(std::is_same_v<decltype(sycl::native::cos(1.0f)), float>);
static_assert(std::is_same_v<decltype(sycl::half_precision::divide(
                                 Vec2f{4.0f, 9.0f}, Vec2f{2.0f, 3.0f})),
                             Vec2f>);
static_assert(std::is_same_v<decltype(sycl::half_precision::divide(
                                 MArray2f{4.0f, 9.0f}, MArray2f{2.0f, 3.0f})),
                             MArray2f>);

// Cover pointer-result scalar builtins through multi_ptr-based signatures.
SYCL_EXTERNAL void
testScalar(sycl::multi_ptr<float, sycl::access::address_space::global_space,
                           sycl::access::decorated::no>
               Ptr) {
  (void)sycl::modf(1.0f, Ptr);
  (void)sycl::sincos(1.0f, Ptr);
}

// Cover a vector pointer-result builtin exposed by the same wrapper.
SYCL_EXTERNAL void
testVector(sycl::multi_ptr<Vec2f, sycl::access::address_space::global_space,
                           sycl::access::decorated::no>
               Ptr) {
  Vec2f Value{1.0f, 2.0f};
  (void)sycl::fract(Value, Ptr);
}

int main() {
  // Instantiate representative calls so the syntax-only compile exercises them.
  Vec2f A{4.0f, 9.0f};
  Vec2f B{2.0f, 3.0f};
  MArray2f MA{4.0f, 9.0f};
  MArray2f MB{2.0f, 3.0f};
  (void)sycl::fmin(1.0f, 2.0f);
  (void)sycl::sqrt(sycl::half{4.0f});
  (void)sycl::fmin(A, B);
  (void)sycl::fmin(MA, MB);
  (void)sycl::native::cos(1.0f);
  (void)sycl::half_precision::divide(A, B);
  (void)sycl::half_precision::divide(MA, MB);
  return 0;
}