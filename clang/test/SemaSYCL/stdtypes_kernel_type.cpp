// RUN: %clang_cc1 -fsycl -fsycl-is-device -sycl-std=2020 -DCHECK_ERROR -verify %s

#include "Inputs/sycl.hpp"

namespace std {
typedef long unsigned int size_t;
typedef long int ptrdiff_t;
typedef decltype(nullptr) nullptr_t;
class T;
class U;
class Foo;
} // namespace std

template <typename T>
struct Templated_kernel_name;

template <typename T>
struct Templated_kernel_name2;

template <typename T, typename... Args> class TemplParamPack;

using namespace cl::sycl;
queue q;

int main() {
#ifdef CHECK_ERROR
  q.submit([&](handler &h) {
    // expected-error@Inputs/sycl.hpp:220 {{invalid type 'nullptr_t' used in kernel name. Type cannot be in the "std" namespace}}
    // expected-note@Inputs/sycl.hpp:220 {{Invalid kernel name is 'nullptr_t'}}
    // expected-note@+1{{in instantiation of function template specialization}}
    h.single_task<std::nullptr_t>([=] {});
  });
  q.submit([&](handler &h) {
    // expected-error@Inputs/sycl.hpp:220 {{invalid type 'std::T' used in kernel name. Type cannot be in the "std" namespace}}
    // expected-note@Inputs/sycl.hpp:220 {{Invalid kernel name is 'std::T'}}
    // expected-note@+1{{in instantiation of function template specialization}}
    h.single_task<std::T>([=] {});
  });
  q.submit([&](handler &h) {
    // expected-error@Inputs/sycl.hpp:220 {{invalid type 'nullptr_t' used in kernel name. Type cannot be in the "std" namespace}}
    // expected-note@Inputs/sycl.hpp:220 {{Invalid kernel name is 'Templated_kernel_name<nullptr_t>'}}
    // expected-note@+1{{in instantiation of function template specialization}}
    h.single_task<Templated_kernel_name<std::nullptr_t>>([=] {});
  });
  q.submit([&](handler &h) {
    // expected-error@Inputs/sycl.hpp:220 {{invalid type 'std::U' used in kernel name. Type cannot be in the "std" namespace}}
    // expected-note@Inputs/sycl.hpp:220 {{Invalid kernel name is 'Templated_kernel_name<std::U>'}}
    // expected-note@+1{{in instantiation of function template specialization}}
    h.single_task<Templated_kernel_name<std::U>>([=] {});
  });
  q.submit([&](handler &cgh) {
    // expected-error@Inputs/sycl.hpp:220 {{invalid type 'std::Foo' used in kernel name. Type cannot be in the "std" namespace}}
    // expected-note@Inputs/sycl.hpp:220{{Invalid kernel name is 'Templated_kernel_name2<Templated_kernel_name<std::Foo>>'}}
    // expected-note@+1{{in instantiation of function template specialization}}
    cgh.single_task<Templated_kernel_name2<Templated_kernel_name<std::Foo>>>([]() {});
  });
  q.submit([&](handler &cgh) {
    // expected-error@Inputs/sycl.hpp:220 {{invalid type 'nullptr_t' used in kernel name. Type cannot be in the "std" namespace}}
    // expected-note@Inputs/sycl.hpp:220 {{Invalid kernel name is 'TemplParamPack<int, float, nullptr_t, double>'}}
    // expected-note@+1{{in instantiation of function template specialization}}
    cgh.single_task<TemplParamPack<int, float, std::nullptr_t, double>>([]() {});
  });
#endif

  // Although in the std namespace, these resolve to builtins such as `int` that are allowed in kernel names
  q.submit([&](handler &h) {
    h.single_task<std::size_t>([=] {});
  });
  q.submit([&](handler &h) {
    h.single_task<std::ptrdiff_t>([=] {});
  });

  return 0;
}
