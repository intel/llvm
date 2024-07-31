// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

int main() {
  bool result = true;
  sycl::queue queue;
  {
    sycl::buffer<bool, 1> res_buf(&result, 1);
    queue.submit([&](sycl::handler &cgh) {
      auto res_acc = res_buf.get_access<sycl::access_mode::read_write>(cgh);

      cgh.parallel_for<class kernel>(sycl::nd_range<3>({1, 1, 1}, {1, 1, 1}),
                                     [=](sycl::nd_item<3> item) {
                                       sycl::sub_group a = item.get_sub_group();

                                       // check for reflexivity
                                       res_acc[0] &= (a == a);
                                       res_acc[0] &= !(a != a);

                                       // check for symmetry
                                       auto copied = a;
                                       auto &b = copied;
                                       res_acc[0] &= (a == b);
                                       res_acc[0] &= (b == a);
                                       res_acc[0] &= !(a != b);
                                       res_acc[0] &= !(b != a);

                                       // check for transitivity
                                       auto copiedTwice = copied;
                                       const auto &c = copiedTwice;
                                       res_acc[0] &= (c == a);
                                     });
    });
    queue.wait_and_throw();
  }

  assert(result);
  return 0;
}
