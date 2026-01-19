// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

constexpr int Val = 42;

int main() {
  sycl::queue Q;
  int OutVal = 0;
  {
    sycl::buffer<int> Buffer(&OutVal, sycl::range{1});
    sycl::accessor Acc(Buffer, sycl::ext::oneapi::accessor_property_list{
                                   sycl::ext::oneapi::no_alias});
    Q.submit([&](sycl::handler &CGH) {
      CGH.require(Acc);
      CGH.single_task([=]() { Acc[0] = Val; });
    });
  }
  assert(OutVal == Val);
  return 0;
}
