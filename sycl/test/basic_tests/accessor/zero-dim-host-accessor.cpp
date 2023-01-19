// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

int main() {
  using DataT = int;
  using AccT = sycl::host_accessor<DataT, 0>;
  int data(5);
  sycl::range<1> r(1);
  sycl::buffer<DataT, 1> data_buf(&data, r);
  AccT acc{data_buf};
  assert(acc.get_size() == sizeof(DataT));
  assert(acc.size() == 1);
  auto ref = &acc;
  assert(*ref == 5);
}
