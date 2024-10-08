// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <iostream>
#include <numeric>
#include <sycl/detail/core.hpp>
#include <sycl/group_algorithm.hpp>
#include <vector>

using namespace sycl;

class back_to_back;

int main() {
  queue q;

  // Use max work-group size to maximize chance of race
  auto KernelID = get_kernel_id<back_to_back>();
  auto KB =
      get_kernel_bundle<bundle_state::executable>(q.get_context(), {KernelID});
  kernel k = KB.get_kernel(KernelID);
  device d = q.get_device();
  int N = k.get_info<sycl::info::kernel_device_specific::work_group_size>(d);

  std::vector<int> Input(N), Sum(N), EScan(N), IScan(N);
  std::iota(Input.begin(), Input.end(), 0);
  std::fill(Sum.begin(), Sum.end(), 0);
  std::fill(EScan.begin(), EScan.end(), 0);
  std::fill(IScan.begin(), IScan.end(), 0);

  {
    buffer<int> InputBuf(Input.data(), N);
    buffer<int> SumBuf(Sum.data(), N);
    buffer<int> EScanBuf(EScan.data(), N);
    buffer<int> IScanBuf(IScan.data(), N);
    q.submit([&](handler &h) {
      auto Input = InputBuf.get_access<access::mode::read>(h);
      auto Sum = SumBuf.get_access<access::mode::write>(h);
      auto EScan = EScanBuf.get_access<access::mode::write>(h);
      auto IScan = IScanBuf.get_access<access::mode::write>(h);
      h.parallel_for<back_to_back>(nd_range<1>(N, N), [=](nd_item<1> it) {
        size_t i = it.get_global_id(0);
        auto g = it.get_group();
        // Loop to increase number of back-to-back calls
        for (int r = 0; r < 10; ++r) {
          Sum[i] = reduce_over_group(g, Input[i], sycl::plus<>());
          EScan[i] = exclusive_scan_over_group(g, Input[i], sycl::plus<>());
          IScan[i] = inclusive_scan_over_group(g, Input[i], sycl::plus<>());
        }
      });
    });
  }

  int sum = 0;
  bool passed = true;
  for (int i = 0; i < N; ++i) {
    passed &= (sum == EScan[i]);
    sum += i;
    passed &= (sum == IScan[i]);
  }
  for (int i = 0; i < N; ++i) {
    passed &= (sum == Sum[i]);
  }

  assert(passed);

  std::cout << "Test passed." << std::endl;
  return 0;
}
