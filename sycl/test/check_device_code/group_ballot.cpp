// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s

#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q;
  Q.parallel_for(sycl::nd_range<1>{32, 32}, [=](sycl::nd_item<1> item) {
    auto Mask = sycl::ext::oneapi::group_ballot(item.get_sub_group());
  });
}
