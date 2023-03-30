// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// NOTE: General tests for atomic fence capabilities.

#include <algorithm>
#include <cassert>
#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;

bool is_supported_order(const std::vector<memory_order> &capabilities,
                        memory_order mem_order) {
  return std::find(capabilities.begin(), capabilities.end(), mem_order) !=
         capabilities.end();
}

bool is_supported_scope(const std::vector<memory_scope> &capabilities,
                        memory_scope mem_scope) {
  return std::find(capabilities.begin(), capabilities.end(), mem_scope) !=
         capabilities.end();
}

void checkFenceBehaviour(memory_order order, memory_scope scope) {
  auto q = queue();
  // Both read and write being release or acquire is wrong. In case order is
  // release or acquire we need read to be acquire and write to be release.
  // If we flip both acquire and release, we will be checking the same case
  // (read == acquire, write == release) twice, so we just skip one case and
  // flip for the other.
  if (order == memory_order::acquire)
    return;
  memory_order order_read = order;
  memory_order order_write = order;
  if (order == memory_order::release) {
    order_read = memory_order::acquire;
  }

  // Count of retries in the check cycle
  constexpr size_t RETRY_COUNT = 256;
  constexpr int expected_val = 42;

  bool res = true;
  int sync = 0;
  int data = 0;
  int value = expected_val;

  // These global_range and local_range values provide a check in one group
  // when test_type = single_group, and between four groups when
  // test_type = between_groups
  range<1> global_range(2);
  range<1> local_range(2);

  {
    buffer<bool> res_buf(&res, range<1>(1));
    buffer<int> sync_buffer(&sync, range<1>(1));
    buffer<int> data_buffer(&data, range<1>(1));
    q.submit([&](handler &cgh) {
      auto res_acc = res_buf.template get_access<access_mode::write>(cgh);
      auto sync_flag_acc =
          sync_buffer.template get_access<sycl::access_mode::read_write>(cgh);
      auto data_acc =
          data_buffer.template get_access<sycl::access_mode::read_write>(cgh);
      cgh.parallel_for(
          nd_range<1>(global_range, local_range), [=](nd_item<1> nditem) {
            atomic_ref<int, memory_order::relaxed, memory_scope::work_group>
                sync_flag(sync_flag_acc[0]);
            int *data = &data_acc[0];
            // Only one nditem should perform non-atomic write.
            // All other nditems should perform non-atomic
            // reads
            if (nditem.get_global_linear_id() == 0) {
              // Non-atomic write to data
              *data = value;
              // Used atomic_fence to guarantee the order
              // instructions execution
              atomic_fence(order_write, scope);
              // Used atomic sync flag to avoid data racing
              sync_flag = 1;
            } else {
              bool write_happened = false;
              for (size_t i = 0; i < RETRY_COUNT; i++) {
                if (sync_flag == 1) {
                  write_happened = true;
                  break;
                }
              }
              atomic_fence(order_read, scope);
              // After the fence safe non-atomic reading
              if (write_happened) {
                // Non-atomic read of data
                if (*data != value)
                  res_acc[0] = false;
              }
            }
          });
    });
  }
  assert(res);
}

int main() {
  queue q;

  std::vector<memory_order> supported_memory_orders =
      q.get_device().get_info<info::device::atomic_fence_order_capabilities>();

  // Relaxed, acquire, release and acq_rel memory order must be supported.
  assert(is_supported_order(supported_memory_orders, memory_order::relaxed));
  assert(is_supported_order(supported_memory_orders, memory_order::acquire));
  assert(is_supported_order(supported_memory_orders, memory_order::release));
  assert(is_supported_order(supported_memory_orders, memory_order::acq_rel));

  std::vector<memory_scope> supported_memory_scopes =
      q.get_device().get_info<info::device::atomic_fence_scope_capabilities>();

  // Work_group, sub_group and work_item memory order must be supported.
  assert(is_supported_scope(supported_memory_scopes, memory_scope::work_item));
  assert(is_supported_scope(supported_memory_scopes, memory_scope::sub_group));
  assert(is_supported_scope(supported_memory_scopes, memory_scope::work_group));

  for (auto order : supported_memory_orders)
    for (auto scope : supported_memory_scopes)
      checkFenceBehaviour(order, scope);

  std::cout << "Test passed." << std::endl;
}
