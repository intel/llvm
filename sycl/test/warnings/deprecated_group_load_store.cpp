// Ignore unexpected warnings because for some reason FE emits the warning
// twice, once for `load`, then for `load<template-params>`.
// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=warning,note %s -o %t.out

#include <sycl/sycl.hpp>

SYCL_EXTERNAL auto test_load(sycl::sub_group sg, int *p) {
  // expected-warning@+1 {{'load' is deprecated: Use sycl::ext::oneapi::experimental::group_load instead.}}
  return sg.load(p);
}

SYCL_EXTERNAL auto test_load(sycl::sub_group sg, sycl::decorated_global_ptr<int> p) {
  // expected-warning@+1 {{'load' is deprecated: Use sycl::ext::oneapi::experimental::group_load instead.}}
  return sg.load(p);
}

SYCL_EXTERNAL auto test_vec_load(sycl::sub_group sg, sycl::decorated_global_ptr<int> p) {
  // expected-warning@+1 {{'load' is deprecated: Use sycl::ext::oneapi::experimental::group_load instead.}}
  return sg.load<4>(p);
}

SYCL_EXTERNAL auto test_store(sycl::sub_group sg, int *p) {
  // expected-warning@+1 {{'store' is deprecated: Use sycl::ext::oneapi::experimental::group_store instead.}}
  return sg.store(p, 42);
}

SYCL_EXTERNAL auto test_vec_store(sycl::sub_group sg, sycl::decorated_global_ptr<int> p) {
  // expected-warning@+1 {{'store' is deprecated: Use sycl::ext::oneapi::experimental::group_store instead.}}
  return sg.store(p, 42);
}

SYCL_EXTERNAL auto test_vec_store(sycl::sub_group sg, sycl::decorated_global_ptr<int> p, sycl::vec<int, 4> v) {
  // expected-warning@+1 {{'store' is deprecated: Use sycl::ext::oneapi::experimental::group_store instead.}}
  return sg.store(p, v);
}
