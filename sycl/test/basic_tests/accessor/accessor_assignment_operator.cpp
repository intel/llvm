// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out

// Testing each of the specialised assignment operators of 0 dimensional
// buffer accessors, local accessors, and host accessors.

#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q;

  {
    int Data = 0;

    using AccT = sycl::accessor<int, 0, sycl::access::mode::read_write>;
    sycl::buffer<int, 1> Buf(&Data, sycl::range<1>(1));

    Q.submit([&](sycl::handler &CGH) {
      sycl::accessor<int, 0> A(Buf, CGH);
      CGH.single_task<class KernelAccessor>([A] {
        typename AccT::value_type DataToWrite = 1;
        A = DataToWrite;
      });

      assert(A == 1);
    });
    Q.wait();
  }

  {
    using AccT = sycl::local_accessor<int, 0>;
    Q.submit([&](sycl::handler &CGH) {
      AccT LA{CGH};
      CGH.single_task<class KernelLocalAccessor>([LA] {
        typename AccT::value_type DataToWrite = 2;
        LA = DataToWrite;
      });

      assert(LA == 2);
    });
    Q.wait();
  }

  {
    int Data = 0;

    using AccT = sycl::host_accessor<int, 0, sycl::access::mode::read_write>;
    sycl::buffer<int> Buf(&Data, sycl::range<1>(1));

    Q.submit([&](sycl::handler &CGH) {
      AccT HA(Buf);
      CGH.single_task<class KernelHostAccessor>([=] {
        typename AccT::value_type DataToWrite = 3;
        HA = DataToWrite;
      });

      assert(HA == 3);
    });
    Q.wait();
  }

  return 0;
}
