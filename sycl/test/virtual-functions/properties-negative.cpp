// RUN: %clangxx %fsycl-host-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s

#include <sycl/sycl.hpp>

namespace oneapi = sycl::ext::oneapi::experimental;

struct user_defined {
  int a;
  float b;
};

int main() {
  sycl::queue q;

  oneapi::properties props_empty{oneapi::indirectly_callable};
  oneapi::properties props_void{oneapi::indirectly_callable_in<void>};
  oneapi::properties props_int{oneapi::indirectly_callable_in<int>};
  oneapi::properties props_user{oneapi::indirectly_callable_in<user_defined>};

  // expected-error-re@sycl/handler.hpp:* {{static assertion failed due to requirement {{.*}} indirectly_callable property cannot be applied to SYCL kernels}}
  q.single_task(props_empty, [=]() {});
  // When both "props_empty" and "props_void" are in use, we won't see the
  // static assert firing for the second one, because there will be only one
  // instantiation of handler::processProperties.
  q.single_task(props_void, [=]() {});
  // expected-error-re@sycl/handler.hpp:* {{static assertion failed due to requirement {{.*}} indirectly_callable property cannot be applied to SYCL kernels}}
  q.single_task(props_int, [=]() {});
  // expected-error-re@sycl/handler.hpp:* {{static assertion failed due to requirement {{.*}} indirectly_callable property cannot be applied to SYCL kernels}}
  q.single_task(props_user, [=]() {});

  return 0;
}
