// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out
#include <cassert>
#include <sycl/sycl.hpp>
using namespace sycl::ext::oneapi::experimental;

std::vector<forward_progress_guarantee> ref1 = {
    forward_progress_guarantee::weakly_parallel};
std::vector<forward_progress_guarantee> ref2 = {
    forward_progress_guarantee::parallel,
    forward_progress_guarantee::concurrent};
std::vector<forward_progress_guarantee> ref3 = {
    forward_progress_guarantee::weakly_parallel,
    forward_progress_guarantee::parallel};
std::vector<forward_progress_guarantee> ref4 = {
    forward_progress_guarantee::weakly_parallel,
    forward_progress_guarantee::parallel,
    forward_progress_guarantee::concurrent};

int main() {
  sycl::queue q;
  sycl::device d = q.get_device();
  auto guarantees = d.get_info<info::device::work_group_progress_capabilities<
      execution_scope::root_group>>();
  assert(guarantees == ref4);
  guarantees = d.get_info<info::device::sub_group_progress_capabilities<
      execution_scope::root_group>>();
  assert(guarantees == ref4);
  guarantees = d.get_info<info::device::sub_group_progress_capabilities<
      execution_scope::work_group>>();
  assert(guarantees == ref4);
  guarantees = d.get_info<info::device::work_item_progress_capabilities<
      execution_scope::root_group>>();
  assert(guarantees == ref1);
  guarantees = d.get_info<info::device::work_item_progress_capabilities<
      execution_scope::sub_group>>();
  assert(guarantees == ref1);
  guarantees = d.get_info<info::device::work_item_progress_capabilities<
      execution_scope::work_group>>();
  assert(guarantees == ref1);
}
