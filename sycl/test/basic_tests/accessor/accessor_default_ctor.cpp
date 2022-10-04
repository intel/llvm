// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out

#include <sycl/sycl.hpp>

int main() {
  sycl::accessor<int, 0, sycl::access::mode::read_write, sycl::target::device>
      B;
  assert(B.empty());
  assert(B.size() == 0);
  assert(B.max_size() == 0);
  assert(B.byte_size() == 0);
  // The return values of get_pointer() and get_multi_ptr() are unspecified.
  assert(B.get_pointer() == nullptr);
  // TODO: uncomment check with get_multi_ptr() when SYCL 2020 mupti_ptr feature
  // will be merged
  // assert(B.get_multi_ptr() == nullptr);

  return 0;
}