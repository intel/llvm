// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -O3 -o %t.out -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_70
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// L0, OpenCL, and HIP backends don't currently support
// info::device::atomic_memory_order_capabilities
// UNSUPPORTED: level_zero, opencl

#include "atomic_memory_order.h"
#include <iostream>
#include <numeric>
using namespace sycl;

void check(queue &q, buffer<int, 2> &res_buf) {
  range<2> size = res_buf.get_range();
  const size_t N_items = 2 * size[0];
  const size_t N_iters = size[1];

  // checking the results is computationally expensive so we do it on the device
  buffer<char, 2> checked_buf(
      {N_items / 2 * N_iters + 1, N_items / 2 * N_iters + 1});

  q.submit([&](handler &cgh) {
    auto res = res_buf.template get_access<access::mode::read>(cgh);
    auto checked =
        checked_buf.template get_access<access::mode::discard_write>(cgh);
    cgh.parallel_for(nd_range<1>(N_items, 32), [=](nd_item<1> it) {
      for (int i = it.get_global_id(0); i < N_items / 2 * N_iters;
           i += N_items) {
        for (int j = 0; j < N_items / 2 * N_iters + 1; j++) {
          checked[i][j] = 0;
        }
      }
    });
  });
  q.submit([&](handler &cgh) {
    auto res = res_buf.template get_access<access::mode::read>(cgh);
    auto checked =
        checked_buf.template get_access<access::mode::discard_write>(cgh);
    cgh.parallel_for(nd_range<1>(N_items / 2, 32), [=](nd_item<1> it) {
      size_t id = it.get_global_id(0);
      for (int i = 1; i < N_iters; i++) {
        for (int j = 0; j < i; j++) {
          if (res[id][j] == res[id][i]) {
            continue;
          }
          checked[res[id][j]][res[id][i]] = 1;
        }
      }
    });
  });
  int err = 0;
  buffer<int> err_buf(&err, 1);
  q.submit([&](handler &cgh) {
    auto res = res_buf.template get_access<access::mode::read>(cgh);
    auto checked = checked_buf.template get_access<access::mode::read>(cgh);
    auto err = err_buf.template get_access<access::mode::discard_write>(cgh);
    cgh.parallel_for(nd_range<1>(N_items / 2, 32), [=](nd_item<1> it) {
      size_t id = it.get_global_id(0);
      for (int i = 1; i < N_iters; i++) {
        for (int j = 0; j < i; j++) {
          if (checked[res[id][i]][res[id][j]]) {
            err[0] = 1;
          }
        }
      }
    });
  });
  auto err_acc = err_buf.get_host_access();
  assert(err_acc[0] == 0);
}

template <memory_order order> void test_global() {
  const size_t N_items = 128;
  const size_t N_iters = 1000;

  int val = 0;

  queue q;
  buffer<int, 2> res_buf({N_items / 2, N_iters});
  buffer<int> val_buf(&val, 1);

  q.submit([&](handler &cgh) {
     auto res = res_buf.template get_access<access::mode::discard_write>(cgh);
     auto val = val_buf.template get_access<access::mode::read_write>(cgh);
     // Intentionally using a small work group size. The assumption being that
     // more sub groups mean more likely failure for the same number of
     // work-items if sequential consistency does not work
     cgh.parallel_for(nd_range<1>(N_items, 16), [=](nd_item<1> it) {
       auto atm = atomic_ref<int, memory_order::acq_rel, memory_scope::device,
                             access::address_space::global_space>(val[0]);
       size_t id = it.get_global_id(0);
       for (int i = 0; i < N_iters; i++) {
         if (id % 2 == 0) {
           atm.store(id / 2 + i * N_items / 2 + 1,
                     order == memory_order::acq_rel ? memory_order::release
                                                    : order);
         } else {
           res[id / 2][i] = atm.load(
               order == memory_order::acq_rel ? memory_order::acquire : order);
         }
       }
     });
   }).wait_and_throw();
  check(q, res_buf);
}

template <memory_order order> void test_local() {
  const size_t N_items = 128;
  const size_t N_iters = 1000;

  int val = 0;

  queue q;
  buffer<int, 2> res_buf({N_items / 2, N_iters});

  q.submit([&](handler &cgh) {
     auto res = res_buf.template get_access<access::mode::discard_write>(cgh);
     local_accessor<int, 1> val(2, cgh);
     cgh.parallel_for(nd_range<1>(N_items, N_items), [=](nd_item<1> it) {
       val[0] = 0;
       it.barrier(access::fence_space::local_space);
       auto atm = atomic_ref<int, memory_order::acq_rel, memory_scope::device,
                             access::address_space::local_space>(val[0]);
       size_t id = it.get_global_id(0);
       for (int i = 0; i < N_iters; i++) {
         if (id % 2 == 0) {
           atm.store(id / 2 + i * N_items / 2 + 1,
                     order == memory_order::acq_rel ? memory_order::release
                                                    : order);
         } else {
           res[id / 2][i] = atm.load(
               order == memory_order::acq_rel ? memory_order::acquire : order);
         }
       }
     });
   }).wait_and_throw();
  check(q, res_buf);
}

int main() {
  queue q;
  std::vector<memory_order> supported_memory_orders =
      q.get_device().get_info<info::device::atomic_memory_order_capabilities>();

  if (!is_supported(supported_memory_orders, memory_order::seq_cst)) {
    std::cout
        << "seq_cst memory order is not supported by the device. Skipping test."
        << std::endl;
    return 0;
  }
  test_global<memory_order::seq_cst>();
  test_local<memory_order::seq_cst>();

  std::cout << "Test passed." << std::endl;
}
