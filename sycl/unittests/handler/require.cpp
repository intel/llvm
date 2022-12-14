#include <gtest/gtest.h>

#include <sycl/sycl.hpp>

TEST(Require, RequireWithNonPlaceholderAccessor) {
  sycl::queue Q;
  int data = 5;
  {
    sycl::buffer<int, 1> buf(&data, 1);
    Q.submit([&](sycl::handler &h) {
      auto acc = buf.get_access<sycl::access::mode::read_write>(h);
      // It should be compilable and does nothing according to the spec
      h.require(acc);
    });
    Q.wait();
  }
}
