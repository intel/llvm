// RUN: %clangxx -fsycl %s -fsyntax-only

#include <sycl/sycl.hpp>

using namespace sycl::ext::oneapi::experimental;

template <forward_progress_guarantee guarantee>
void check_props(sycl::queue &q) {
  q.single_task(
      properties{work_group_progress<guarantee, execution_scope::root_group>},
      [=]() {});
  q.single_task(
      properties{sub_group_progress<guarantee, execution_scope::root_group>},
      [=]() {});
  q.single_task(
      properties{sub_group_progress<guarantee, execution_scope::work_group>},
      [=]() {});
  q.single_task(
      properties{work_item_progress<guarantee, execution_scope::root_group>},
      [=]() {});
  q.single_task(
      properties{work_item_progress<guarantee, execution_scope::work_group>},
      [=]() {});
  q.single_task(
      properties{work_item_progress<guarantee, execution_scope::sub_group>},
      [=]() {});
}

int main() {
  sycl::queue q;
  check_props<forward_progress_guarantee::parallel>(q);
  check_props<forward_progress_guarantee::weakly_parallel>(q);
  check_props<forward_progress_guarantee::concurrent>(q);
}
