// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  queue q;
  auto *RedMem = malloc_shared<int>(1, q);
  auto *Success = malloc_shared<bool>(1, q);
  *Success = true;

  *RedMem = 0;
  q.parallel_for(range<1>{7}, reduction(RedMem, std::plus<int>{}),
                 [=](item<1> Item, auto &Red) {
                   Red += 1;
                   if (Item.get_range(0) != 7)
                     *Success = false;
                   if (Item.get_id(0) == 7)
                     *Success = false;
                 })
      .wait();

  assert(*RedMem == 7);
  assert(*Success);

  *RedMem = 0;
  q.parallel_for(range<2>{1030, 7}, reduction(RedMem, std::plus<int>{}),
                 [=](item<2> Item, auto &Red) {
                   Red += 1;
                   if (Item.get_range(0) != 1030)
                     *Success = false;
                   if (Item.get_range(1) != 7)
                     *Success = false;

                   if (Item.get_id(0) == 1030)
                     *Success = false;
                   if (Item.get_id(1) == 7)
                     *Success = false;
                 })
      .wait();

  assert(*RedMem == 1030 * 7);
  assert(*Success);

  free(RedMem, q);
  free(Success, q);
  return 0;
}
