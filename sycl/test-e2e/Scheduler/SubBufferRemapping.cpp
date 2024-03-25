// RUN: %{build} -o %t.out
// RUN: env SYCL_HOST_UNIFIED_MEMORY=1 env SYCL_PI_TRACE=2  %{run} %t.out | FileCheck %s

// sub-buffer host alloca are not mated with device alloca. That linkage occurs
// in the parent alloca. this test ensures that any map operations are using the
// correct alloca, even in the case of sub-buffer accessors in host tasks.

// CHECK: == fills completed
// CHECK: piEnqueueMemBufferMap
// CHECK: piEnqueueMemBufferMap
// CHECK-NEXT: <unknown> :
// CHECK-NEXT: pi_mem :
// CHECK-NEXT: <unknown> :
// CHECK-NEXT: <unknown> : 3

// CHECK: == between host accesses
// CHECK: piEnqueueMemBufferMap
// CHECK-NEXT: <unknown> :
// CHECK-NEXT: pi_mem :
// CHECK-NEXT: <unknown> :
// CHECK-NEXT: <unknown> : 3

#include <sycl/detail/core.hpp>

int main(int argc, const char **argv) {

  sycl::queue q;

  {
    sycl::range<1> bufRange(25);
    sycl::buffer<double, 1> buf_b(bufRange);
    q.submit([&](sycl::handler &cgh) {
      auto acc_buf_b = buf_b.get_access(cgh, sycl::write_only);
      cgh.fill<double>(acc_buf_b, 1.0);
    });

    sycl::buffer<double, 1> buf_y(buf_b.get_range());
    q.submit([&](sycl::handler &cgh) {
      auto acc_buf_y = buf_y.get_access(cgh, sycl::write_only);
      cgh.fill<double>(acc_buf_y, -1.0);
    });

    q.wait();
    std::cout << "== fills completed" << std::endl;

    sycl::buffer<double, 1> buf_b_sub(buf_b, 0, buf_b.get_range());

    q.submit([&](sycl::handler &cgh) {
      auto acc_b_sub = buf_b_sub.get_host_access(cgh, sycl::read_only);
      auto acc_buf_y = buf_y.get_host_access(cgh, sycl::read_write);

      cgh.host_task([=]() { acc_buf_y[0] = 1.0 - acc_b_sub[0]; });
    });
    q.wait();

    std::cout << "== between host accesses" << std::endl;

    q.submit([&](sycl::handler &cgh) {
      auto acc_buf_y = buf_y.get_host_access(cgh, sycl::read_only);
      auto acc_b_sub = buf_b_sub.get_host_access(cgh, sycl::read_write);

      cgh.host_task([=]() { acc_b_sub[0] = 1.0 + acc_buf_y[0]; });
    });
    q.wait();
  }

  return 0;
}
