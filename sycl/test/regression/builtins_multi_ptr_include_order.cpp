// RUN: %clangxx -fsycl -fsyntax-only -Wno-deprecated-declarations %s -DTEST_BUILTINS_ONLY
// RUN: %clangxx -fsycl -fsyntax-only -Wno-deprecated-declarations %s -DTEST_BUILTINS_FIRST
// RUN: %clangxx -fsycl -fsyntax-only -Wno-deprecated-declarations %s -DTEST_MULTI_PTR_FIRST

// Regression coverage for builtins/multi_ptr decoupling.
// We want to preserve these behaviors:
// 1. <sycl/builtins.hpp> compiles without including <sycl/multi_ptr.hpp>.
// 2. Including builtins before multi_ptr still allows later multi_ptr
//    instantiation for scalar pointer builtins.
// 3. Including builtins before multi_ptr still allows later multi_ptr
//    instantiation for vector pointer builtins.
// 4. Including multi_ptr before builtins also works for those builtin calls.

#if defined(TEST_BUILTINS_ONLY)
#include <sycl/builtins.hpp>

int main() {
  auto Value = sycl::fmin(1.0f, 2.0f);
  (void)Value;
  return 0;
}

#elif defined(TEST_BUILTINS_FIRST)
#include <sycl/builtins.hpp>
#include <sycl/multi_ptr.hpp>

SYCL_EXTERNAL void
testScalar(sycl::multi_ptr<float, sycl::access::address_space::global_space,
                           sycl::access::decorated::no>
               Ptr) {
  (void)sycl::modf(1.0f, Ptr);
  (void)sycl::sincos(1.0f, Ptr);
}

SYCL_EXTERNAL void
testVector(sycl::multi_ptr<sycl::vec<float, 2>,
                           sycl::access::address_space::global_space,
                           sycl::access::decorated::no>
               Ptr) {
  sycl::vec<float, 2> Value{1.0f, 2.0f};
  (void)sycl::fract(Value, Ptr);
}

int main() { return 0; }

#elif defined(TEST_MULTI_PTR_FIRST)
// clang-format off
#include <sycl/multi_ptr.hpp>
#include <sycl/builtins.hpp>
// clang-format on

SYCL_EXTERNAL void
testScalar(sycl::multi_ptr<float, sycl::access::address_space::global_space,
                           sycl::access::decorated::no>
               Ptr) {
  (void)sycl::modf(1.0f, Ptr);
  (void)sycl::sincos(1.0f, Ptr);
}

SYCL_EXTERNAL void
testVector(sycl::multi_ptr<sycl::vec<float, 2>,
                           sycl::access::address_space::global_space,
                           sycl::access::decorated::no>
               Ptr) {
  sycl::vec<float, 2> Value{1.0f, 2.0f};
  (void)sycl::fract(Value, Ptr);
}

int main() { return 0; }
#endif