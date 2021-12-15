// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -verify %s

// Diagnostic tests for device_has(aspect, ...) attribute

#include "sycl.hpp"

namespace fake_cl {
namespace sycl {
enum class aspect {
  aspect1,
  aspect2
};
}
} // namespace fake_cl

[[sycl::device_has()]] int a; // expected-error{{'device_has' attribute only applies to functions}}

[[sycl::device_has("123")]] void func1() {}                          // expected-error{{invalid argument for 'device_has' attribute; argument must be device aspect of type cl::sycl::aspect}}
[[sycl::device_has(fake_cl::sycl::aspect::aspect1)]] void func2() {} // expected-error{{invalid argument for 'device_has' attribute; argument must be device aspect of type cl::sycl::aspect}}

[[sycl::device_has(cl::sycl::aspect::cpu)]] void func3();   // expected-note{{previous attribute is here}}
[[sycl::device_has(cl::sycl::aspect::gpu)]] void func3() {} // expected-warning{{attribute 'device_has' is already applied}}

template <fake_cl::sycl::aspect Aspect>
[[sycl::device_has(Aspect)]] void func4() {} // expected-error 2{{invalid argument for 'device_has' attribute; argument must be device aspect of type cl::sycl::aspect}}

void checkTemplate() {
  func4<fake_cl::sycl::aspect::aspect1>(); // expected-note {{in instantiation of function template specialization 'func4<fake_cl::sycl::aspect::aspect1>' requested here}}
}

[[sycl::device_has(1)]] void func5() {} // expected-error{{invalid argument for 'device_has' attribute; argument must be device aspect of type cl::sycl::aspect}}

template <typename Ty>
[[sycl::device_has(Ty{})]] void func6() {} // expected-error{{invalid argument for 'device_has' attribute; argument must be device aspect of type cl::sycl::aspect}}
