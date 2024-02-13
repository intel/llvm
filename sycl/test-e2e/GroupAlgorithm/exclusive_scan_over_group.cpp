// CPU and ACC not yet supported:
// Unsupported SPIR-V module SPIRV module requires unsupported capability 6400
// REQUIRES: gpu

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <algorithm>
#include <iostream>
#include <sycl/sycl.hpp>

template <typename T, typename AccessorT>
sycl::event compiler_group_scan_impl(sycl::queue *queue, AccessorT &in_data,
                                     AccessorT &out_data, int num_wg,
                                     int group_size) {
  sycl::nd_range<1> thread_range(num_wg * group_size, group_size);
  sycl::event event = queue->submit([&](sycl::handler &cgh) {
    cgh.require(in_data);
    cgh.require(out_data);
    cgh.parallel_for(thread_range, [=](sycl::nd_item<1> item) {
      auto id = item.get_global_linear_id();
      auto group = item.get_group();
      T data = in_data[id];

      T updated_data =
          sycl::exclusive_scan_over_group(group, data, sycl::multiplies<T>());
      out_data[id] = updated_data;
    });
  });
  return event;
}

template <typename T, typename AccessorT>
void test_compiler_group_scan(sycl::queue *queue, AccessorT &in_data,
                              AccessorT &out_data, int num_wg, int group_size) {
  compiler_group_scan_impl<T>(queue, in_data, out_data, num_wg, group_size);
}

int main(int argc, const char **argv) {
  constexpr int num_wg = 1;
  constexpr int group_size = 16;

  sycl::queue queue;
  constexpr size_t nelems = num_wg * group_size;
  int data[nelems];
  int result[nelems];
  for (size_t i = 0; i < nelems; ++i) {
    data[i] = 2;
    result[i] = 0;
  }
  sycl::buffer<int> data_buf{&data[0], sycl::range{nelems}};
  sycl::buffer<int> result_buf{&result[0], sycl::range{nelems}};
  sycl::accessor data_acc{data_buf};
  sycl::accessor result_acc{result_buf};
  test_compiler_group_scan<int>(&queue, data_acc, result_acc, num_wg,
                                group_size);
  sycl::host_accessor result_host{result_buf};
  int expected[] = {1,   2,   4,    8,    16,   32,   64,    128,
                    256, 512, 1024, 2048, 4096, 8192, 16384, 32768};
  for (int i = 0; i < sizeof(expected) / sizeof(int); ++i) {
    assert(result_host[i] == expected[i]);
  }
  return 0;
}
