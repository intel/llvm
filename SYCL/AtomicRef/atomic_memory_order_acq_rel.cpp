// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -O3 -o %t.out -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_70
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// L0, OpenCL, and HIP backends don't currently support
// info::device::atomic_memory_order_capabilities
// UNSUPPORTED: level_zero, opencl, hip

// NOTE: Tests fetch_add for acquire and release memory ordering.

#include "atomic_memory_order.h"
#include <iostream>
#include <numeric>
using namespace sycl;

template <memory_order order> void test_acquire_global() {
  const size_t N_items = 1024;
  const size_t N_iters = 1000;

  int error = 0;
  int val[] = {0, 0};

  queue q;
  {
    buffer<int> error_buf(&error, 1);
    buffer<int> val_buf(val, 1);

    q.submit([&](handler &cgh) {
       auto error =
           error_buf.template get_access<access::mode::read_write>(cgh);
       auto val = val_buf.template get_access<access::mode::read_write>(cgh);
       cgh.parallel_for(range<1>(N_items), [=](item<1> it) {
         volatile int *val_p = val.get_pointer();
         auto atm0 =
             atomic_ref<int, memory_order::relaxed, memory_scope::device,
                        access::address_space::global_space>(val[0]);
         auto atm1 =
             atomic_ref<int, memory_order::relaxed, memory_scope::device,
                        access::address_space::global_space>(val[1]);
         for (int i = 0; i < N_iters; i++) {
           if (it.get_id(0) == 0) {
             atm0.fetch_add(1, order);
             val_p[1]++;
           } else {
             int tmp1 = atm1.load(memory_order::acquire);
             int tmp0 = atm0.load(memory_order::relaxed);
             if (tmp0 < tmp1) {
               error[0] = 1;
             }
           }
         }
       });
     }).wait_and_throw();
  }
  assert(error == 0);
}

template <memory_order order> void test_acquire_local() {
  const size_t local_size = 1024;
  const size_t N_wgs = 16;
  const size_t global_size = local_size * N_wgs;
  const size_t N_iters = 1000;

  int error = 0;
  int val[] = {0, 0};

  queue q;
  {
    buffer<int> error_buf(&error, 1);
    buffer<int> val_buf(val, 1);

    q.submit([&](handler &cgh) {
       auto error =
           error_buf.template get_access<access::mode::read_write>(cgh);
       local_accessor<int, 1> val(2, cgh);
       cgh.parallel_for(
           nd_range<1>(global_size, local_size), [=](nd_item<1> it) {
             size_t lid = it.get_local_id(0);
             val[0] = 0;
             val[1] = 0;
             it.barrier(access::fence_space::local_space);
             volatile int *val_p = val.get_pointer();
             auto atm0 =
                 atomic_ref<int, memory_order::relaxed, memory_scope::device,
                            access::address_space::local_space>(val[0]);
             auto atm1 =
                 atomic_ref<int, memory_order::relaxed, memory_scope::device,
                            access::address_space::local_space>(val[1]);
             for (int i = 0; i < N_iters; i++) {
               if (it.get_local_id(0) == 0) {
                 atm0.fetch_add(1, order);
                 val_p[1]++;
               } else {
                 int tmp1 = atm1.load(memory_order::acquire);
                 int tmp0 = atm0.load(memory_order::relaxed);
                 if (tmp0 < tmp1) {
                   error[0] = 1;
                 }
               }
             }
           });
     }).wait_and_throw();
  }
  assert(error == 0);
}

template <memory_order order> void test_release_global() {
  const size_t N_items = 1024;
  const size_t N_iters = 1000;

  int error = 0;
  int val[] = {0, 0};

  queue q;
  {
    buffer<int> error_buf(&error, 1);
    buffer<int> val_buf(val, 1);

    q.submit([&](handler &cgh) {
       auto error =
           error_buf.template get_access<access::mode::read_write>(cgh);
       auto val = val_buf.template get_access<access::mode::read_write>(cgh);
       cgh.parallel_for(range<1>(N_items), [=](item<1> it) {
         volatile int *val_p = val.get_pointer();
         auto atm0 =
             atomic_ref<int, memory_order::relaxed, memory_scope::device,
                        access::address_space::global_space>(val[0]);
         auto atm1 =
             atomic_ref<int, memory_order::relaxed, memory_scope::device,
                        access::address_space::global_space>(val[1]);
         for (int i = 0; i < N_iters; i++) {
           if (it.get_id(0) == 0) {
             val_p[0]++;
             atm1.fetch_add(1, order);
           } else {
             int tmp1 = atm1.load(memory_order::acquire);
             int tmp0 = atm0.load(memory_order::relaxed);
             if (tmp0 < tmp1) {
               error[0] = 1;
             }
           }
         }
       });
     }).wait_and_throw();
  }
  assert(error == 0);
}

template <memory_order order> void test_release_local() {
  const size_t local_size = 1024;
  const size_t N_wgs = 16;
  const size_t global_size = local_size * N_wgs;
  const size_t N_iters = 1000;

  int error = 0;
  int val[] = {0, 0};

  queue q;
  {
    buffer<int> error_buf(&error, 1);
    buffer<int> val_buf(val, 1);

    q.submit([&](handler &cgh) {
       auto error =
           error_buf.template get_access<access::mode::read_write>(cgh);
       local_accessor<int, 1> val(2, cgh);
       cgh.parallel_for(
           nd_range<1>(global_size, local_size), [=](nd_item<1> it) {
             size_t lid = it.get_local_id(0);
             val[0] = 0;
             val[1] = 0;
             it.barrier(access::fence_space::local_space);
             volatile int *val_p = val.get_pointer();
             auto atm0 =
                 atomic_ref<int, memory_order::relaxed, memory_scope::device,
                            access::address_space::local_space>(val[0]);
             auto atm1 =
                 atomic_ref<int, memory_order::relaxed, memory_scope::device,
                            access::address_space::local_space>(val[1]);
             for (int i = 0; i < N_iters; i++) {
               if (it.get_local_id(0) == 0) {
                 val_p[0]++;
                 atm1.fetch_add(1, order);
               } else {
                 int tmp1 = atm1.load(memory_order::acquire);
                 int tmp0 = atm0.load(memory_order::relaxed);
                 if (tmp0 < tmp1) {
                   error[0] = 1;
                 }
               }
             }
           });
     }).wait_and_throw();
  }
  assert(error == 0);
}

int main() {
  queue q;
  std::vector<memory_order> supported_memory_orders =
      q.get_device().get_info<info::device::atomic_memory_order_capabilities>();

  if (is_supported(supported_memory_orders, memory_order::acquire)) {
    std::cout << "Testing acquire" << std::endl;
    test_acquire_global<memory_order::acquire>();
    test_acquire_local<memory_order::acquire>();
  }
  if (is_supported(supported_memory_orders, memory_order::release)) {
    std::cout << "Testing release" << std::endl;
    test_release_global<memory_order::release>();
    test_release_local<memory_order::release>();
  }
  if (is_supported(supported_memory_orders, memory_order::acq_rel)) {
    std::cout << "Testing acq_rel" << std::endl;
    // Acquire-release memory order must also support both acquire and release
    // orderings.
    assert(is_supported(supported_memory_orders, memory_order::acquire) &&
           is_supported(supported_memory_orders, memory_order::release));
    test_acquire_global<memory_order::acq_rel>();
    test_acquire_local<memory_order::acq_rel>();
    test_release_global<memory_order::acq_rel>();
    test_release_local<memory_order::acq_rel>();
  }
  if (is_supported(supported_memory_orders, memory_order::seq_cst)) {
    std::cout << "Testing seq_cst" << std::endl;
    test_acquire_global<memory_order::seq_cst>();
    test_acquire_local<memory_order::seq_cst>();
    test_release_global<memory_order::seq_cst>();
    test_release_local<memory_order::seq_cst>();
  }

  std::cout << "Test passed." << std::endl;
}
