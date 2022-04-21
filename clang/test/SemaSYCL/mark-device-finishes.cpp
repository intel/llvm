// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -verify -fsyntax-only -std=c++20 %s

// This test validates that this actually makes it through 'MarkDevice'.  This
// is a bit of a pathological case where we ended up visiting each call
// individually. There is likely a similar test case that can cause us to hit
// a pathological case in a very similar situation (where the callees aren't
// exactly the same), but that likely causes problems with template
// instantiations first.

// expected-no-diagnostics

#include "sycl.hpp"

template<bool B, typename V = void>
struct enable_if { };
template<typename V>
struct enable_if<true, V> {
  using type = V;
};
template<bool B, typename V = void>
using enable_if_t = typename enable_if<B, V>::type;


template<int N, enable_if_t<N == 24, int> = 0>
void mark_device_pathological_case() {
  // Do nothing.
}

template<int N, enable_if_t<N < 24, int> = 0>
void mark_device_pathological_case() {
  // We were visiting each of these, which caused 9^24 visits.
  mark_device_pathological_case<N + 1>();
  mark_device_pathological_case<N + 1>();
  mark_device_pathological_case<N + 1>();
  mark_device_pathological_case<N + 1>();
  mark_device_pathological_case<N + 1>();
  mark_device_pathological_case<N + 1>();
  mark_device_pathological_case<N + 1>();
  mark_device_pathological_case<N + 1>();
  mark_device_pathological_case<N + 1>();
  mark_device_pathological_case<N + 1>();
}

int main() {
  sycl::queue q;
  q.submit([](sycl::handler &h) {
    h.single_task<class kernel>([]() { mark_device_pathological_case<0>(); });
  });
}
