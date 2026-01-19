// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=warning,note %s

#include <sycl/detail/core.hpp>

#include <sycl/ext/oneapi/work_group_static.hpp>

using namespace sycl::ext::oneapi::experimental;

class InvalidCtorDtor {
  InvalidCtorDtor() {}
  ~InvalidCtorDtor() {}
};

SYCL_EXTERNAL void test(int *p) {
  // expected-error-re@sycl/ext/oneapi/work_group_static.hpp:* {{static assertion failed due to requirement {{.+}}: Can only be used with non const and non volatile types}}
  sycl::ext::oneapi::experimental::work_group_static<const volatile int> b1;
  // expected-error-re@sycl/ext/oneapi/work_group_static.hpp:* {{static assertion failed due to requirement {{.+}}: Can only be used with trivially constructible and destructible types}}
  sycl::ext::oneapi::experimental::work_group_static<InvalidCtorDtor> b2;
}
