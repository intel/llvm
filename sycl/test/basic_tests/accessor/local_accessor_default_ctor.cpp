// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

int main ()
{
  using AccT = sycl::local_accessor<int, 1>;
  bool empty;
  size_t byte_size;
  size_t size;
  size_t max_size;

  AccT acc;
  empty = acc.empty();
  byte_size = acc.byte_size();
  size = acc.size();
  max_size = acc.max_size();
  // The return values of get_pointer() and get_multi_ptr() are
  // unspecified. Just check they can run without any issue.
  auto ptr = acc.get_pointer();
  // TODO: uncomment check with get_multi_ptr() when SYCL 2020 multi_ptr feature
  // will be merged
  // auto multi_ptr = acc.get_multi_ptr();

  assert(empty == true);
  assert(byte_size == 0);
  assert(size == 0);
  assert(max_size == 0);

  return 0;
}
