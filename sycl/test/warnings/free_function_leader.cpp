// RUN: %clangxx %fsycl-host-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s -o %t.out

#include <sycl/sycl.hpp>

int main() {
  using namespace sycl;
  queue q;
  q.parallel_for(nd_range<1>{range<1>{42}, range<1>{42}}, [=](nd_item<1> item) {
     // expected-warning@+1 {{ext::oneapi::leader free function is deprecated. Use member function leader of the sycl::group/sycl::sub_group instead.}}
     std::ignore = leader(item.get_sub_group());
   }).wait();
  return 0;
}
