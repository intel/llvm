// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out

#include <sycl/sycl.hpp>

int main() {
  std::vector<size_t> res(5);
  sycl::buffer<size_t> bufRes(res.data(), res.size());
  sycl::queue testQueue;

  sycl::accessor<int, 0, sycl::access::mode::read_write, sycl::target::device>
      B;
  assert(B.empty());
  assert(B.size() == 0);
  assert(B.max_size() == 0);
  assert(B.byte_size() == 0);
  assert(B.get_pointer() == nullptr);

  return 0;
}