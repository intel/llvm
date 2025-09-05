// RUN: %clangxx %s -fsycl-device-only -Xclang -verify %s -Xclang -verify-ignore-unexpected=note,warning

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;

// expected-error@* {{group_load requires a pointer if alignment property is used}}
SYCL_EXTERNAL void test(sycl::sub_group &sg,
                        sycl::detail::accessor_iterator<int, 1> accessor_iter,
                        int &out) {
  group_load(sg, accessor_iter, out, properties(alignment<16>));
}

// expected-error@* {{group_store requires a pointer if alignment property is used}}
SYCL_EXTERNAL void test(sycl::sub_group &sg,
                        sycl::detail::accessor_iterator<int, 1> accessor_iter,
                        int v) {
  group_store(sg, v, accessor_iter, properties(alignment<16>));
}
