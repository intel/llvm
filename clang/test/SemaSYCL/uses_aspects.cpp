// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -verify %s

// Diagnostic tests for __uses_aspects__(aspect, ...) attribute

#include "sycl.hpp"

namespace fake_cl {
namespace sycl {
enum class aspect {
  aspect1,
  aspect2
};
}
} // namespace fake_cl

[[__sycl_detail__::__uses_aspects__()]] int a; // expected-error{{'__uses_aspects__' attribute only applies to classes and functions}}

[[__sycl_detail__::__uses_aspects__("123")]] void func1() {}                          // expected-error{{'__uses_aspects__' attribute argument is invalid; argument must be device aspect of type sycl::aspect}}
[[__sycl_detail__::__uses_aspects__(fake_cl::sycl::aspect::aspect1)]] void func2() {} // expected-error{{'__uses_aspects__' attribute argument is invalid; argument must be device aspect of type sycl::aspect}}

[[__sycl_detail__::__uses_aspects__(cl::sycl::aspect::cpu)]] void func3();   // expected-note{{previous attribute is here}}
[[__sycl_detail__::__uses_aspects__(cl::sycl::aspect::gpu)]] void func3() {} // expected-warning{{attribute '__uses_aspects__' is already applied}}

template <fake_cl::sycl::aspect Aspect>
[[__sycl_detail__::__uses_aspects__(Aspect)]] void func4() {} // expected-error 2{{'__uses_aspects__' attribute argument is invalid; argument must be device aspect of type sycl::aspect}}

void checkTemplate() {
  func4<fake_cl::sycl::aspect::aspect1>(); // expected-note {{in instantiation of function template specialization 'func4<fake_cl::sycl::aspect::aspect1>' requested here}}
}

[[__sycl_detail__::__uses_aspects__(1)]] void func5() {} // expected-error{{'__uses_aspects__' attribute argument is invalid; argument must be device aspect of type sycl::aspect}}

template <typename Ty>
[[__sycl_detail__::__uses_aspects__(Ty{})]] void func6() {} // expected-error{{'__uses_aspects__' attribute argument is invalid; argument must be device aspect of type sycl::aspect}}

[[__sycl_detail__::__uses_aspects__(cl::sycl::aspect::cpu)]] // expected-note{{previous attribute is here}}
[[__sycl_detail__::__uses_aspects__(cl::sycl::aspect::gpu)]] void
func7() {} // expected-warning@-1{{attribute '__uses_aspects__' is already applied}}
