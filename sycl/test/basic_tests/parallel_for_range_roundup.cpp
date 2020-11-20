// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple  %s -o %t.out
// RUN: env SYCL_PARALLEL_FOR_RANGE_ROUNDING_TRACE=1 %GPU_RUN_PLACEHOLDER %t.out %GPU_CHECK_PLACEHOLDER
// RUN: env SYCL_PARALLEL_FOR_RANGE_ROUNDING_TRACE=1 %CPU_RUN_PLACEHOLDER %t.out %CPU_CHECK_PLACEHOLDER

#include <CL/sycl.hpp>

using namespace sycl;

class PF_init_item;
class PF_init_id;
class PF_init_id1;

struct SizesInfo {
  range<1> ItemGlobalSize = {0};
  range<1> RealGlobalSizeX = {0};
  range<1> RealGlobalSizeY = {0};
  range<1> RealGlobalSizeZ = {0};
};

void check(const char *msg, size_t v, size_t ref) {
  std::cout << msg << v << std::endl;
  assert(v == ref);
}

int try_item(size_t size) {
  size_t RoundedUpSize = (size + 32 - 1) / 32 * 32;
  SizesInfo SInfo;
  range<1> Size{size};
  int Counter = 0;

  {
    buffer<SizesInfo, 1> BufSizes(&SInfo, 1);
    buffer<int, 1> BufCounter(&Counter, 1);
    queue myQueue;

    myQueue.submit([&](handler &cgh) {
      auto AccSizes = BufSizes.get_access<access::mode::read_write>(cgh);
      auto AccCounter = BufCounter.get_access<access::mode::atomic>(cgh);

      cgh.parallel_for<PF_init_item>(Size, [=](item<1> ITEM) {
        AccCounter[0].fetch_add(1);
        AccSizes[0].ItemGlobalSize = ITEM.get_range(0);
      });
    });
    myQueue.wait();
  }

  check("Size seen by user = ", SInfo.ItemGlobalSize.get(0), size);
  check("Counter = ", Counter, size);
  return 0;
}

int try_id(size_t size) {
  size_t RoundedUpSize = (size + 32 - 1) / 32 * 32;
  SizesInfo SInfo;
  range<1> Size{size};
  int Counter = 0;
  bool OnGpu;

  {
    buffer<SizesInfo, 1> BufSizes(&SInfo, 1);
    buffer<int, 1> BufCounter(&Counter, 1);
    queue myQueue;

    myQueue.submit([&](handler &cgh) {
      auto AccSizes = BufSizes.get_access<access::mode::read_write>(cgh);
      auto AccCounter = BufCounter.get_access<access::mode::atomic>(cgh);

      cgh.parallel_for<class PF_init_id>(Size, [=](id<1> ID) {
        AccCounter[0].fetch_add(1);
        AccSizes[0].ItemGlobalSize = ID[0];
      });
    });
    myQueue.wait();
  }
  check("Counter = ", Counter, size);

  {
    buffer<SizesInfo, 1> BufSizes(&SInfo, 1);
    buffer<int, 1> BufCounter(&Counter, 1);
    queue myQueue;
    Counter = 0;

    myQueue.submit([&](handler &cgh) {
      auto AccSizes = BufSizes.get_access<access::mode::read_write>(cgh);
      auto AccCounter = BufCounter.get_access<access::mode::atomic>(cgh);
      cgh.parallel_for<class PF_init_id1>(Size, [=](id<1> ID) {
        AccCounter[0].fetch_add(1);
        AccSizes[0].ItemGlobalSize = ID[0];
      });
    });
    myQueue.wait();
  }
  check("Counter = ", Counter, size);

  return 0;
}

int main() {
  int x;

  x = 10;
  try_item(x);
  try_id(x);

  x = 256;
  try_item(x);
  try_id(x);

  return 0;
}

// CHECK:       parallel_for range adjusted from 10 to 32
// CHECK-NEXT:  Size seen by user = 10
// CHECK-NEXT:  Counter = 10
// CHECK-NEXT:  parallel_for range adjusted from 10 to 32
// CHECK-NEXT:  Counter = 10
// CHECK-NEXT:  parallel_for range adjusted from 10 to 32
// CHECK-NEXT:  Counter = 10
// CHECK-NEXT:  Size seen by user = 256
// CHECK-NEXT:  Counter = 256
// CHECK-NEXT:  Counter = 256
// CHECK-NEXT:  Counter = 256
