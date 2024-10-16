// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

#include <cassert>

using namespace sycl::ext::oneapi::experimental;

std::vector<forward_progress_guarantee> ref1 = {
    forward_progress_guarantee::weakly_parallel};
std::vector<forward_progress_guarantee> ref2 = {
    forward_progress_guarantee::weakly_parallel,
    forward_progress_guarantee::parallel,
    forward_progress_guarantee::concurrent};

int main() {
  sycl::queue q;
  sycl::device d = q.get_device();

  // Check guarantees for execution_scope::root_group
  auto guarantees = d.get_info<info::device::work_group_progress_capabilities<
      execution_scope::root_group>>();
  assert(guarantees == ref2);

  guarantees = d.get_info<info::device::sub_group_progress_capabilities<
      execution_scope::root_group>>();
  assert(guarantees == ref2);

  guarantees = d.get_info<info::device::work_item_progress_capabilities<
      execution_scope::root_group>>();
  assert(guarantees == ref1);

  // Check guarantees for execution_scope::work_group
  guarantees = d.get_info<info::device::sub_group_progress_capabilities<
      execution_scope::work_group>>();
  assert(guarantees == ref2);

  guarantees = d.get_info<info::device::work_item_progress_capabilities<
      execution_scope::work_group>>();
  assert(guarantees == ref1);

  // Check guarantees for execution_scope::sub_group
  guarantees = d.get_info<info::device::work_item_progress_capabilities<
      execution_scope::sub_group>>();
  assert(guarantees == ref1);
}
