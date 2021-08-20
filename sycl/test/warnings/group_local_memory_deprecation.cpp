// RUN: %clangxx -fsycl-device-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s -o %t.out

#include <CL/sycl.hpp>

class KernelA;

int main() {
  sycl::queue Q;
  Q.submit([&](sycl::handler &Cgh) {
    Cgh.parallel_for<KernelA>(
        sycl::nd_range<1>(sycl::range<1>(1), sycl::range<1>(1)),
        [=](sycl::nd_item<1> Item) {
          sycl::multi_ptr<int, sycl::access::address_space::local_space> GlmA =
              // expected-warning@+1 {{'group_local_memory<int, sycl::group<1>, int>' is deprecated: use sycl::ext::oneapi::group_local_memory instead}}
              sycl::group_local_memory<int>(Item.get_group(), 1);
          sycl::multi_ptr<int, sycl::access::address_space::local_space> GlmB =
              // expected-warning@+1 {{'group_local_memory_for_overwrite<int, sycl::group<1>>' is deprecated: use sycl::ext::oneapi::group_local_memory_for_overwrite instead}}
              sycl::group_local_memory_for_overwrite<int>(Item.get_group());
        });
  });

  return 0;
}
