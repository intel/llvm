// REQUIRES: xptifw, opencl
// RUN: %clangxx %s -DXPTI_COLLECTOR -DXPTI_CALLBACK_API_EXPORTS %xptifw_lib %shared_lib %fPIC %cxx_std_optionc++17 -o %t_collector.dll
// RUN: %clangxx -fsycl -O2 %s -o %t.opt.out
// RUN: env XPTI_TRACE_ENABLE=1 XPTI_FRAMEWORK_DISPATCHER=%xptifw_dispatcher XPTI_SUBSCRIBERS=%t_collector.dll %BE_RUN_PLACEHOLDER %t.opt.out | FileCheck %s --check-prefix=CHECK-OPT
// RUN: %clangxx -fsycl -fno-sycl-dead-args-optimization %s -o %t.noopt.out
// RUN: env XPTI_TRACE_ENABLE=1 XPTI_FRAMEWORK_DISPATCHER=%xptifw_dispatcher XPTI_SUBSCRIBERS=%t_collector.dll %BE_RUN_PLACEHOLDER %t.noopt.out | FileCheck %s --check-prefix=CHECK-NOOPT

#ifdef XPTI_COLLECTOR

#include "../Inputs/buffer_info_collector.cpp"

#else
#include <array>
#include <numeric>
#include <sycl/sycl.hpp>

using namespace sycl;
int main() {
  std::array<int, 1024> input;
  std::iota(input.begin(), input.end(), 0);
  buffer<int> valuesBuf{input};

  // Buffers with just 1 element to get the reduction results
  int sumResult = 0;
  buffer<int> sumBuf{&sumResult, 1};
  queue myQueue;
  myQueue.submit([&](handler &cgh) {
    auto inputValues = valuesBuf.get_access<access_mode::read>(cgh);

    auto sumR = reduction(sumBuf, cgh, plus<>());
    // Reduction kernel is used
    // CHECK-OPT:Node create|{{.*}}reduction{{.*}}test1{{.*}}|{{.*}}.cpp:[[# @LINE - 5 ]]:3|{1024, 1, 1}, {{{.*}}, 1, 1}, {0, 0, 0}, 5
    // CHECK-NOOPT:Node create|{{.*}}reduction{{.*}}test1{{.*}}|{{.*}}.cpp:[[# @LINE - 6 ]]:3|{1024, 1, 1}, {{{.*}}, 1, 1}, {0, 0, 0}, 13
    cgh.parallel_for<class test1>(
        range<1>{1024}, sumR,
        [=](id<1> idx, auto &sum) { sum += inputValues[idx]; });
  });

  // sumBuf contains the reduction results once the kernel completes
  assert(sumBuf.get_host_access()[0] == 523776);

  {
    buffer<int> in_buf(input.data(), input.size());
    buffer<int> out_buf(input.size());
    myQueue.submit([&](handler &cgh) {
      auto in = in_buf.template get_access<access::mode::read>(cgh);
      auto out = out_buf.template get_access<access::mode::read_write>(cgh);
      // CHECK-OPT:Node create|{{.*}}test2{{.*}}|{{.*}}.cpp:[[# @LINE - 3 ]]:5|{128, 4, 2}, {32, 2, 1}, {16, 1, 0}, 2
      // CHECK-NOOPT:Node create|{{.*}}test2{{.*}}|{{.*}}.cpp:[[# @LINE - 4 ]]:5|{128, 4, 2}, {32, 2, 1}, {16, 1, 0}, 8
      cgh.parallel_for<class test2>(
          nd_range<3>({128, 4, 2}, {32, 2, 1}, {16, 1, 0}), [=](nd_item<3> it) {
            auto sg = it.get_sub_group();
            joint_exclusive_scan(sg, in.get_pointer(),
                                 in.get_pointer() + sg.get_local_id(),
                                 out.get_pointer(), std::plus<>{});
          });
    });
  }
}
#endif
