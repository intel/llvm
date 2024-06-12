// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// The purpose of this test is to check that the forward_progress_guarantee
// properties associated with a kernel are compiled, submitted and verified
// successfully during runtime for a gpu device of the level_zero backend. In
// this context, verification during runtime means for the runtime to verify
// that the device to which a kernel is submitted actually provides the
// guarantees required by the kernel submission. If the device does not provide
// all guarantees, then we verify that an exception is thrown as written in the
// spec of the forward progress extension.

#include <sycl/detail/core.hpp>

#include <cassert>

using namespace sycl::ext::oneapi::experimental;

// Primary template
template <forward_progress_guarantee guarantee>
void check_props(sycl::queue &q) {}

// Full specializations for each progress guarantee

template <>
void check_props<forward_progress_guarantee::parallel>(sycl::queue &q) {
  constexpr auto guarantee = forward_progress_guarantee::parallel;
  // Check properties at execution_scope::root_group coordination level
  q.single_task(
      properties{work_group_progress<guarantee, execution_scope::root_group>},
      [=]() {});
  q.single_task(
      properties{sub_group_progress<guarantee, execution_scope::root_group>},
      [=]() {});
  try {
    q.single_task(
        properties{work_item_progress<guarantee, execution_scope::root_group>},
        [=]() {});
    assert(false && "Expected exception not seen!");
  } catch (sycl::exception &ex) {
  }

  // Check properties at execution_scope::work_group coordination level
  q.single_task(
      properties{sub_group_progress<guarantee, execution_scope::work_group>},
      [=]() {});
  try {
    q.single_task(
        properties{work_item_progress<guarantee, execution_scope::work_group>},
        [=]() {});
    assert(false && "Expected exception not seen!");
  } catch (sycl::exception &ex) {
  }

  // Check properties at execution_scope::sub_group coordination level
  try {
    q.single_task(
        properties{work_item_progress<guarantee, execution_scope::sub_group>},
        [=]() {});
  } catch (sycl::exception &ex) {
  }
}

template <>
void check_props<forward_progress_guarantee::weakly_parallel>(sycl::queue &q) {
  constexpr auto guarantee = forward_progress_guarantee::weakly_parallel;
  // Check properties at execution_scope::root_group coordination level
  q.single_task(
      properties{work_group_progress<guarantee, execution_scope::root_group>},
      [=]() {});
  q.single_task(
      properties{sub_group_progress<guarantee, execution_scope::root_group>},
      [=]() {});

  q.single_task(
      properties{work_item_progress<guarantee, execution_scope::root_group>},
      [=]() {});

  // Check properties at execution_scope::work_group coordination level
  q.single_task(
      properties{sub_group_progress<guarantee, execution_scope::work_group>},
      [=]() {});
  q.single_task(
      properties{work_item_progress<guarantee, execution_scope::work_group>},
      [=]() {});

  // Check properties at execution_scope::sub_group coordination level
  q.single_task(
      properties{work_item_progress<guarantee, execution_scope::sub_group>},
      [=]() {});
}

template <>
void check_props<forward_progress_guarantee::concurrent>(sycl::queue &q) {
  constexpr auto guarantee = forward_progress_guarantee::concurrent;
  // Check properties at execution_scope::root_group coordination level
  q.single_task(
      properties{work_group_progress<guarantee, execution_scope::root_group>},
      [=]() {});
  q.single_task(
      properties{sub_group_progress<guarantee, execution_scope::root_group>},
      [=]() {});
  try {
    q.single_task(
        properties{work_item_progress<guarantee, execution_scope::root_group>},
        [=]() {});
    assert(false && "Expected exception not seen!");
  } catch (sycl::exception &ex) {
  }

  // Check properties at execution_scope::work_group coordination level
  q.single_task(
      properties{sub_group_progress<guarantee, execution_scope::work_group>},
      [=]() {});
  try {
    q.single_task(
        properties{work_item_progress<guarantee, execution_scope::work_group>},
        [=]() {});
    assert(false && "Expected exception not seen!");
  } catch (sycl::exception &ex) {
  }

  // Check properties at execution_scope::sub_group coordination level
  try {
    q.single_task(
        properties{work_item_progress<guarantee, execution_scope::sub_group>},
        [=]() {});
    assert(false && "Expected exception not seen!");
  } catch (sycl::exception &ex) {
  }
}

int main() {
  sycl::queue q;
  check_props<forward_progress_guarantee::parallel>(q);
  check_props<forward_progress_guarantee::weakly_parallel>(q);
  check_props<forward_progress_guarantee::concurrent>(q);
}
