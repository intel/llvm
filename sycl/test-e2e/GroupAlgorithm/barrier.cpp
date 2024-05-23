// Temporarily disabled due to regressions introduced by
// https://github.com/intel/llvm/pull/8412.
// REQUIRES: TEMPORARY_DISABLED

// RUN: %{build} -o %t.out -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80
// RUN: %{run} %t.out

// REQUIRES: aspect-ext_oneapi_cuda_async_barrier
// REQUIRES: cuda

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/cuda/barrier.hpp>
#include <vector>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::cuda;

void basic() {
  queue q{};
  int N = 64;
  std::vector<int> data(N);
  for (int i = 0; i < N; i++) {
    data[i] = i;
  }
  {
    buffer<int> buf(data.data(), N);

    q.submit([&](handler &cgh) {
      auto acc = buf.get_access<access::mode::read_write>(cgh);
      local_accessor<int, 1> loc(N, cgh);
      local_accessor<barrier, 1> loc_barrier(2, cgh);
      cgh.parallel_for(nd_range<1>(N, N), [=](nd_item<1> item) {
        size_t idx = item.get_local_linear_id();
        loc[idx] = acc[idx];
        if (idx < 2) {
          loc_barrier[idx].initialize(N);
        }
        item.barrier(access::fence_space::local_space);
        for (int i = 0; i < N; i++) {
          int val = loc[idx];
          barrier::arrival_token arr = loc_barrier[0].arrive();
          val += 1;
          int dst_idx = (idx + 1) % N;
          loc_barrier[0].wait(arr);
          loc[dst_idx] = val;
          loc_barrier[1].wait(loc_barrier[1].arrive());
        }
        acc[idx] = loc[idx];
      });
    });
  }
  for (int i = 0; i < N; i++) {
    assert(data[i] == i + N);
  }
}

void interface() {
  queue q{};
  int N = 64;
  std::vector<int> data(N, -1);
  std::vector<int> test1(N, -1);
  std::vector<int> test2(N, -1);
  for (int i = 0; i < N; i++) {
    data[i] = i;
  }
  {
    buffer<int> data_buf(data.data(), N);
    buffer<int> test1_buf(test1.data(), N);
    buffer<int> test2_buf(test2.data(), N);

    q.submit([&](handler &cgh) {
      auto data_acc = data_buf.get_access<access::mode::read_write>(cgh);
      auto test1_acc = test1_buf.get_access<access::mode::read_write>(cgh);
      auto test2_acc = test2_buf.get_access<access::mode::read_write>(cgh);
      local_accessor<int, 1> loc(N, cgh);
      local_accessor<barrier, 1> loc_barrier(2, cgh);
      cgh.parallel_for(nd_range<1>(N, N), [=](nd_item<1> item) {
        size_t idx = item.get_local_linear_id();
        if (idx == 0) {
          loc_barrier[0].initialize(N);
        }
        if (idx == 1) {
          loc_barrier[1].initialize(N * N);
        }
        item.barrier(access::fence_space::local_space);

        item.async_work_group_copy(
            loc.get_multi_ptr<access::decorated::yes>(),
            data_acc.get_multi_ptr<access::decorated::yes>(), N);
        loc_barrier[1].arrive_copy_async();
        barrier::arrival_token arr = loc_barrier[1].arrive_no_complete(N - 1);
        loc_barrier[1].arrive_and_wait();

        if (idx == 0) {
          loc_barrier[0].invalidate();
          int *reused_barrier_space = (int *)(void *)loc_barrier.get_pointer();
          *reused_barrier_space = loc[0];
          loc[0] = 0;
        }
        item.barrier(access::fence_space::local_space);
        if (idx == 1) {
          int *reused_barrier_space = (int *)(void *)loc_barrier.get_pointer();
          loc[0] = *reused_barrier_space;
        }
        item.barrier(access::fence_space::local_space);
        if (idx == 0) {
          loc_barrier[0].initialize(N);
        }

        int val = loc[idx];
        arr = loc_barrier[0].arrive();
        val = (val + 1) % N;
        int dst_idx = (idx + 1) % N;
        loc_barrier[0].wait(arr);
        loc[dst_idx] = val;
        loc_barrier[0].wait(loc_barrier[0].arrive());

        item.async_work_group_copy(
            data_acc.get_multi_ptr<access::decorated::yes>(),
            loc.get_multi_ptr<access::decorated::yes>(), N);
        loc_barrier[1].arrive_copy_async_no_inc();
        loc_barrier[1].arrive_no_complete(N - 3);
        arr = loc_barrier[1].arrive();
        test1_acc[idx] = loc_barrier[1].test_wait(arr);
        arr = loc_barrier[1].arrive();
        item.barrier(access::fence_space::local_space);
        test2_acc[idx] = loc_barrier[1].test_wait(arr);
        loc_barrier[1].wait(arr);

        loc_barrier[1].arrive_no_complete(N - 6);
        loc_barrier[1].arrive_and_drop_no_complete(5);
        arr = loc_barrier[1].arrive_and_drop();
        loc_barrier[1].wait(arr);

        for (int i = 0; i < N - 6; i++) {
          arr = loc_barrier[1].arrive();
        }
        loc_barrier[1].wait(arr);
      });
    });
  }
  for (int i = 0; i < N; i++) {
    assert(data[i] == i);
    assert(test1[i] == 0);
    assert(test2[i] == 1);
  }
}

int main() {
  queue q;
  basic();
  interface();

  return 0;
}
