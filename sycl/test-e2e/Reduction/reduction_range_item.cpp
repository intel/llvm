// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Windows doesn't yet have full shutdown().
// UNSUPPORTED: ze_debug && windows
#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  queue q;
  auto *RedMem = malloc_device<int>(1, q);
  auto *Success = malloc_device<bool>(1, q);
  int RedMemHost;
  bool SuccessHost;
  RedMemHost = 0;
  SuccessHost = true;
  q.memcpy(RedMem, &RedMemHost, sizeof(int)).wait();
  q.memcpy(Success, &SuccessHost, sizeof(bool)).wait();
  q.parallel_for(range<1>{7}, reduction(RedMem, std::plus<int>{}),
                 [=](item<1> Item, auto &Red) {
                   Red += 1;
                   if (Item.get_range(0) != 7)
                     *Success = false;
                   if (Item.get_id(0) == 7)
                     *Success = false;
                 })
      .wait();
  q.memcpy(&RedMemHost, RedMem, sizeof(int)).wait();
  q.memcpy(&SuccessHost, Success, sizeof(bool)).wait();
  assert(RedMemHost == 7);
  assert(SuccessHost);

  RedMemHost = 0;
  q.memcpy(RedMem, &RedMemHost, sizeof(int)).wait();
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

  q.memcpy(&RedMemHost, RedMem, sizeof(int)).wait();
  q.memcpy(&SuccessHost, Success, sizeof(bool)).wait();
  assert(RedMemHost == 1030 * 7);
  assert(SuccessHost);

  free(RedMem, q);
  free(Success, q);
  return 0;
}
