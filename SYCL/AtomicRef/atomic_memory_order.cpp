// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// L0, OpenCL, and ROCm backends don't currently support
// info::device::atomic_memory_order_capabilities and aspect::atomic64
// XFAIL: level_zero || opencl || rocm

// NOTE: Tests load and store for supported memory orderings.

#include <CL/sycl.hpp>
#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>
using namespace sycl;
using namespace sycl::ONEAPI;

template <typename T, memory_order MO> class memory_order_kernel;

template <typename T> void acq_rel_test(queue q, size_t N) {
  T a = 0;
  {
    buffer<T> a_buf(&a, 1);

    q.submit([&](handler &cgh) {
      auto a_acc = a_buf.template get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<memory_order_kernel<T, memory_order::acq_rel>>(
          range<1>(N), [=](item<1> it) {
            int gid = it.get_id(0);
            auto aar =
                atomic_ref<T, memory_order::acq_rel, memory_scope::device,
                           access::address_space::global_space>(a_acc[0]);
            auto ld = aar.load();
            ld += 1;
            aar.store(ld);
          });
    });
  }

  // All work-items increment by 1, so final value should be equal to N
  assert(a == T(N));
}

template <typename T> void seq_cst_test(queue q, size_t N) {
  T a = 0;
  T b = 0;
  {
    buffer<T> a_buf(&a, 1);
    buffer<T> b_buf(&b, 1);

    q.submit([&](handler &cgh) {
      auto a_acc = a_buf.template get_access<access::mode::read_write>(cgh);
      auto b_acc = b_buf.template get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<memory_order_kernel<T, memory_order::seq_cst>>(
          range<1>(N), [=](item<1> it) {
            int gid = it.get_id(0);
            auto aar =
                atomic_ref<T, memory_order::seq_cst, memory_scope::device,
                           access::address_space::global_space>(a_acc[0]);
            auto bar =
                atomic_ref<T, memory_order::seq_cst, memory_scope::device,
                           access::address_space::global_space>(b_acc[0]);
            auto ald = aar.load();
            auto bld = bar.load();
            ald += 1;
            bld += ald;
            bar.store(bld);
            aar.store(ald);
          });
    });
  }

  // All work-items increment a by 1, so final value should be equal to N
  assert(a == T(N));
  // b is the sum of [1..N]
  size_t rsum = 0;
  for (size_t i = 1; i <= N; ++i)
    rsum += i;
  assert(b == T(rsum));
}

bool is_supported(std::vector<memory_order> capabilities,
                  memory_order mem_order) {
  return std::find(capabilities.begin(), capabilities.end(), mem_order) !=
         capabilities.end();
}

int main() {
  queue q;

  std::vector<memory_order> supported_memory_orders =
      q.get_device().get_info<info::device::atomic_memory_order_capabilities>();
  bool atomic64_support = q.get_device().has(aspect::atomic64);

  constexpr int N = 32;

  // Relaxed memory order must be supported. This ordering is used in other
  // tests.
  assert(is_supported(supported_memory_orders, memory_order::relaxed));

  if (is_supported(supported_memory_orders, memory_order::acq_rel)) {
    // Acquire-release memory order must also support both acquire and release
    // orderings.
    assert(is_supported(supported_memory_orders, memory_order::acquire) &&
           is_supported(supported_memory_orders, memory_order::release));
    acq_rel_test<int>(q, N);
    acq_rel_test<unsigned int>(q, N);
    acq_rel_test<float>(q, N);
    if (sizeof(long) == 4) {
      // long is 32-bit
      acq_rel_test<long>(q, N);
      acq_rel_test<unsigned long>(q, N);
    }
    if (atomic64_support) {
      if (sizeof(long) == 8) {
        // long is 64-bit
        acq_rel_test<long>(q, N);
        acq_rel_test<unsigned long>(q, N);
      }
      acq_rel_test<long long>(q, N);
      acq_rel_test<unsigned long long>(q, N);
      acq_rel_test<double>(q, N);
    }
  }

  if (is_supported(supported_memory_orders, memory_order::seq_cst)) {
    seq_cst_test<int>(q, N);
    seq_cst_test<unsigned int>(q, N);
    seq_cst_test<float>(q, N);
    if (sizeof(long) == 4) {
      // long is 32-bit
      seq_cst_test<long>(q, N);
      seq_cst_test<unsigned long>(q, N);
    }
    if (atomic64_support) {
      if (sizeof(long) == 8) {
        // long is 64-bit
        seq_cst_test<long>(q, N);
        seq_cst_test<unsigned long>(q, N);
      }
      seq_cst_test<long long>(q, N);
      seq_cst_test<unsigned long long>(q, N);
      seq_cst_test<double>(q, N);
    }
  }

  std::cout << "Test passed." << std::endl;
}
