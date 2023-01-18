// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out

// Testing each of the specialised assignment operators of 0 dimensional
// buffer accessors, local accessors, and host accessors.

#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;

  {
    int data = 16;

    sycl::buffer<int, 1> Buf(&data, sycl::range<1>(1));

    q.submit([&](sycl::handler &CGH) {
      sycl::accessor<int, 0> A(Buf, CGH);
      CGH.single_task<class KernelAccessor>([A] { A = 1; });

      assert(A == 1);
    })
    .wait_and_throw();
  }

  q.submit([&](sycl::handler &CGH) {
    sycl::local_accessor<int, 0> LA{CGH};
    CGH.single_task<class KernelLocalAccessor>([LA] { LA = 2; });

    assert(LA == 2);
  })
  .wait_and_throw();

  {
    using AccT = sycl::accessor<int, 0, sycl::access::mode::read_write, sycl::access::target::host_buffer>;
    int data = 16;

    sycl::buffer<int> Buf(&data, sycl::range<1>(1));

    q.submit([&](sycl::handler &CGH) {
      AccT HA(Buf, CGH);
      CGH.single_task<class KernelHostAccessor>([=] {
        typename AccT::value_type Data = 3;
        HA = Data;
      });

      assert(HA == 3);
    })
    .wait_and_throw();
  }

  return 0;
}