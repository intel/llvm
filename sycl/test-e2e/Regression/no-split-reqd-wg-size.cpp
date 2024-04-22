// This test checks that with -fsycl-device-code-split=off, two kernels
// with different reqd_work_group_size dimensions can be launched.
// RUN: %{build} -fsycl -fsycl-device-code-split=off -o %t.out
// RUN: %{run} %t.out
#include <sycl/sycl.hpp>

constexpr int WGSIZE = 4;

using namespace sycl;

void kernel_launch_2(queue &q) {
  range<1> globalRange(WGSIZE);
  range<1> localRange(WGSIZE);
  nd_range<1> NDRange(globalRange, localRange);
  q.submit([&](handler &cgh) {
     cgh.parallel_for<class testNDRange2>(
         NDRange, [=](nd_item<1> it) [[sycl::reqd_work_group_size(WGSIZE)]] {});
   }).wait();
}

void kernel_launch(queue &q) {
  range<2> globalRange(WGSIZE, WGSIZE);
  range<2> localRange(WGSIZE, WGSIZE);
  nd_range<2> NDRange(globalRange, localRange);
  q.submit([&](handler &cgh) {
     cgh.parallel_for<class testNDRange>(
         NDRange,
         [=](nd_item<2> it) [[sycl::reqd_work_group_size(WGSIZE, WGSIZE)]] {});
   }).wait();
}

int main(int argc, char **argv) {
  queue q;

  kernel_launch_2(q);
  kernel_launch(q);

  return 0;
}
