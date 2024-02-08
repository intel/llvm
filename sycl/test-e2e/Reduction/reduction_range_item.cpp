// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Windows doesn't yet have full shutdown().
// UNSUPPORTED: ze_debug && windows

#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  queue q;
  bool Success_val = true;
  int RedMem_val = 0;
  sycl::buffer RedMem_buf{&RedMem_val, sycl::range{1}};
  sycl::buffer Success_buf{&Success_val, sycl::range{1}};
  q.submit([&](sycl::handler &h) {
    sycl::accessor RedMem{RedMem_buf, h};
    sycl::accessor Success{Success_buf, h};
    h.parallel_for(range<1>{7}, reduction(RedMem, std::plus<int>{}),
                   [=](item<1> Item, auto &Red) {
                     Red += 1;
                     if (Item.get_range(0) != 7)
                       Success[0] = false;
                     if (Item.get_id(0) == 7)
                       Success[0] = false;
                   });
  });
  sycl::host_accessor RedMem{RedMem_buf};
  sycl::host_accessor Success{Success_buf};
  assert(RedMem[0] == 7);
  assert(Success[0]);
  RedMem[0] = 0;
  q.submit([&](sycl::handler &h) {
    sycl::accessor RedMem{RedMem_buf, h};
    sycl::accessor Success{Success_buf, h};
    h.parallel_for(range<2>{1030, 7}, reduction(RedMem, std::plus<int>{}),
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
                   });
  });
  sycl::host_accessor RedMem{RedMem_buf};
  sycl::host_accessor Success{Success_buf};

  assert(RedMem[0] == 1030 * 7);
  assert(Success[0]);

  return 0;
}
