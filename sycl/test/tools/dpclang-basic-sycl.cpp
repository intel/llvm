// Verify that the dpclang++ binary can compile basic SYCL code.
// RUN: %dpclangxx -fsycl -fsycl-targets=%sycl_triple -I %sycl_include -c %s -o %t.o

#include <sycl/sycl.hpp>

void test() {
  sycl::queue q;
  int result = 0;
  sycl::buffer<int> buf(&result, 1);

  q.submit([&](sycl::handler &h) {
    sycl::accessor acc(buf, h, sycl::write_only);
    h.single_task([=]() { acc[0] = 42; });
  });
}
