// RUN: %clang_cc1 -fsycl-is-device -sycl-std=2020 -DCHECK_ERROR -verify %s

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
    // expected-error@#KernelSingleTask {{'nullptr_t' is an invalid kernel name type}}
    // expected-note@#KernelSingleTask {{type 'nullptr_t' cannot be in the "std" namespace}}
    // expected-note@+1{{in instantiation of function template specialization}}
    h.single_task<std::nullptr_t>([=] {});
  });
  q.submit([&](handler &h) {
    // expected-error@#KernelSingleTask {{'std::T' is an invalid kernel name type}}
    // expected-note@#KernelSingleTask {{type 'std::T' cannot be in the "std" namespace}}
    // expected-note@+1{{in instantiation of function template specialization}}
    h.single_task<std::T>([=] {});
  });
  q.submit([&](handler &h) {
    // expected-error@#KernelSingleTask {{'Templated_kernel_name<nullptr_t>' is an invalid kernel name type}}
    // expected-note@#KernelSingleTask {{type 'nullptr_t' cannot be in the "std" namespace}}
    // expected-note@+1{{in instantiation of function template specialization}}
    h.single_task<Templated_kernel_name<std::nullptr_t>>([=] {});
  });
  q.submit([&](handler &h) {
    // expected-error@#KernelSingleTask {{'Templated_kernel_name<std::U>' is an invalid kernel name type}}
    // expected-note@#KernelSingleTask {{type 'std::U' cannot be in the "std" namespace}}
    // expected-note@+1{{in instantiation of function template specialization}}
    h.single_task<Templated_kernel_name<std::U>>([=] {});
  });
  q.submit([&](handler &cgh) {
    // expected-error@#KernelSingleTask {{'Templated_kernel_name2<Templated_kernel_name<std::Foo>>' is an invalid kernel name type}}
    // expected-note@#KernelSingleTask{{type 'std::Foo' cannot be in the "std" namespace}}
    // expected-note@+1{{in instantiation of function template specialization}}
    cgh.single_task<Templated_kernel_name2<Templated_kernel_name<std::Foo>>>([]() {});
  });
  q.submit([&](handler &cgh) {
    // expected-error@#KernelSingleTask {{'TemplParamPack<int, float, nullptr_t, double>' is an invalid kernel name type}}
    // expected-note@#KernelSingleTask {{type 'nullptr_t' cannot be in the "std" namespace}}
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
