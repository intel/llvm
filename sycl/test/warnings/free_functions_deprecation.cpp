// RUN: %clangxx %fsycl-host-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s -o %t.out

#include <sycl/sycl.hpp>

int main() {
  namespace sycl_exp = sycl::ext::oneapi::experimental;
  // expected-warning@+1 {{'this_nd_item<1>' is deprecated: use sycl::ext::oneapi::this_work_item::get_nd_item() instead}}
  (void)sycl_exp::this_nd_item<1>();
  // expected-warning@+1 {{'this_group<1>' is deprecated: use sycl::ext::oneapi::this_work_item::get_work_group() instead}}
  (void)sycl_exp::this_group<1>();
  // expected-warning@+1 {{'this_sub_group' is deprecated: use sycl::ext::oneapi::this_work_item::get_sub_group() instead}}
  (void)sycl_exp::this_sub_group();

  // expected-warning@+1 {{'this_item<1>' is deprecated: use nd_range kernel and sycl::ext::oneapi::this_work_item::get_nd_item() instead}}
  (void)sycl_exp::this_item<1>();
  // expected-warning@+1 {{'this_id<1>' is deprecated: use nd_range kernel and sycl::ext::oneapi::this_work_item::get_nd_item() instead}}
  (void)sycl_exp::this_id<1>();

  // expected-warning@+1 {{'get_root_group<1>' is deprecated: use sycl::ext::oneapi::experimental::this_work_item::get_root_group() instead}}
  (void)sycl_exp::this_kernel::get_root_group<1>();

  return 0;
}
