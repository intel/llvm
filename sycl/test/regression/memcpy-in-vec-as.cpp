// RUN: %clangxx -fsycl -D_FORTIFY_SOURCE=2 %s -o %t.out
#include <iostream>
#include <sycl/sycl.hpp>

int main() {
  using res_vec_type = sycl::vec<sycl::cl_ushort, 4>;
  res_vec_type res;

  sycl::vec<sycl::cl_uchar, 8> RefData(1, 2, 3, 4, 5, 6, 7, 8);
  {
    sycl::buffer<res_vec_type, 1> OutBuf(&res, sycl::range<1>(1));
    sycl::buffer<decltype(RefData), 1> InBuf(&RefData, sycl::range<1>(1));
    sycl::queue Queue;
    Queue.submit([&](sycl::handler &cgh) {
      auto In = InBuf.get_access<sycl::access::mode::read>(cgh);
      auto Out = OutBuf.get_access<sycl::access::mode::write>(cgh);
      cgh.single_task<class as_op>(
          [=]() { Out[0] = In[0].as<res_vec_type>(); });
    });
  }

  if (res.s0() != 513 || res.s1() != 1027 || res.s2() != 1541 || res.s3() != 2055) {
    std::cerr << "Incorrect result" << std::endl;
    return 1;
  }

  return 0;
}
