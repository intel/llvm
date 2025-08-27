// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -verify -DUSR -fsyntax-only %s
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -verify -fsyntax-only %s
// UNSUPPORTED: system-windows
// This test checks that an undefined "__glibcxx_assert_fail" without SYCL_EXTERNAL will lead to SYCL sema check
// failure if it is not declared in a system header otherwise no SYCL sema check failure will be triggered.

#include "sycl.hpp"
#ifdef USR
namespace std {
void __glibcxx_assert_fail(const char*, int, const char*, const char*) noexcept;
// expected-note@-1 {{'__glibcxx_assert_fail' declared here}}
}
#else
#include <dummy_failed_assert>
#endif

#ifdef USR
SYCL_EXTERNAL
void call_glibcxx_assert_fail() {
  // expected-note@-1 {{called by 'call_glibcxx_assert_fail'}}
  std::__glibcxx_assert_fail("file1", 10, "func1", "dummy test");
  // expected-error@-1 {{SYCL kernel cannot call an undefined function without SYCL_EXTERNAL attribute}}
}
#else
// expected-no-diagnostics
SYCL_EXTERNAL
void call_glibcxx_assert_fail() {
  std::__glibcxx_assert_fail("file1", 10, "func1", "dummy test");
}
#endif

