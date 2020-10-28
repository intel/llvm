// RUN: %clang_cc1 -fsycl -fsycl-is-device -sycl-std=2020 -DCHECK_ERROR -verify %s

#include "Inputs/sycl.hpp"

namespace std {
typedef long unsigned int size_t;
typedef long int ptrdiff_t;
typedef decltype(nullptr) nullptr_t;
class T;
class U;
} // namespace std

template <typename T>
struct Templated_kernel_name;

template <typename T, typename... Args> class TemplParamPack;

using namespace cl::sycl;
queue q;

int main() {
#ifdef CHECK_ERROR
  // expected-error@Inputs/sycl.hpp:220 5 {{kernel name cannot be a type in the "std" namespace}}
  q.submit([&](handler &h) {
    // expected-note@+1{{in instantiation of function template specialization}}
    h.single_task<std::nullptr_t>([=] {});
  });
  q.submit([&](handler &h) {
    // expected-note@+1{{in instantiation of function template specialization}}
    h.single_task<std::T>([=] {});
  });
  q.submit([&](handler &h) {
    // expected-note@+1{{in instantiation of function template specialization}}
    h.single_task<Templated_kernel_name<std::nullptr_t>>([=] {});
  });
  q.submit([&](handler &h) {
    // expected-note@+1{{in instantiation of function template specialization}}
    h.single_task<Templated_kernel_name<std::U>>([=] {});
  });
  q.submit([&](handler &cgh) {
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
