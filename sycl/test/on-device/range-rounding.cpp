// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple  %s -o %t.out
// REQUIRES: cpu
// RUN: env SYCL_PARALLEL_FOR_RANGE_ROUNDING_TRACE=1 %CPU_RUN_PLACEHOLDER %t.out %CPU_CHECK_PLACEHOLDER

#include <CL/sycl.hpp>

using namespace sycl;

range<1> Range1 = {0};
range<2> Range2 = {0, 0};
range<3> Range3 = {0, 0, 0};

void check(const char *msg, size_t v, size_t ref) {
  std::cout << msg << v << std::endl;
  assert(v == ref);
}

int try_item1(size_t size) {
  range<1> Size{size};
  int Counter = 0;
  {
    buffer<range<1>, 1> BufRange(&Range1, 1);
    buffer<int, 1> BufCounter(&Counter, 1);
    queue myQueue;

    myQueue.submit([&](handler &cgh) {
      auto AccRange = BufRange.get_access<access::mode::read_write>(cgh);
      auto AccCounter = BufCounter.get_access<access::mode::atomic>(cgh);
      cgh.parallel_for<class PF_init_item1>(Size, [=](item<1> ITEM) {
        AccCounter[0].fetch_add(1);
        AccRange[0] = this_item<1>().get_range(0);
      });
    });
    myQueue.wait();
  }
  check("Size seen by user = ", Range1.get(0), size);
  check("Counter = ", Counter, size);
  return 0;
}

void try_item2(size_t size) {
  range<2> Size{size, size};
  int Counter = 0;
  {
    buffer<range<2>, 1> BufRange(&Range2, 1);
    buffer<int, 1> BufCounter(&Counter, 1);
    queue myQueue;

    myQueue.submit([&](handler &cgh) {
      auto AccRange = BufRange.get_access<access::mode::read_write>(cgh);
      auto AccCounter = BufCounter.get_access<access::mode::atomic>(cgh);
      cgh.parallel_for<class PF_init_item2>(Size, [=](item<2> ITEM) {
        AccCounter[0].fetch_add(1);
        AccRange[0][0] = this_item<2>().get_range(0);
      });
    });
    myQueue.wait();
  }
  check("Size seen by user = ", Range2.get(0), size);
  check("Counter = ", Counter, size * size);
}

void try_item3(size_t size) {
  range<3> Size{size, size, size};
  int Counter = 0;
  {
    buffer<range<3>, 1> BufRange(&Range3, 1);
    buffer<int, 1> BufCounter(&Counter, 1);
    queue myQueue;

    myQueue.submit([&](handler &cgh) {
      auto AccRange = BufRange.get_access<access::mode::read_write>(cgh);
      auto AccCounter = BufCounter.get_access<access::mode::atomic>(cgh);
      cgh.parallel_for<class PF_init_item3>(Size, [=](item<3> ITEM) {
        AccCounter[0].fetch_add(1);
        AccRange[0][0] = this_item<3>().get_range(0);
      });
    });
    myQueue.wait();
  }
  check("Size seen by user = ", Range3.get(0), size);
  check("Counter = ", Counter, size * size * size);
}

void try_id1(size_t size) {
  range<1> Size{size};
  int Counter = 0;
  {
    buffer<range<1>, 1> BufRange(&Range1, 1);
    buffer<int, 1> BufCounter(&Counter, 1);
    queue myQueue;

    myQueue.submit([&](handler &cgh) {
      auto AccRange = BufRange.get_access<access::mode::read_write>(cgh);
      auto AccCounter = BufCounter.get_access<access::mode::atomic>(cgh);
      cgh.parallel_for<class PF_init_id1>(Size, [=](id<1> ID) {
        AccCounter[0].fetch_add(1);
        AccRange[0] = this_id<1>()[0];
      });
    });
    myQueue.wait();
  }
  check("Counter = ", Counter, size);
}

void try_id2(size_t size) {
  range<2> Size{size, size};
  int Counter = 0;
  {
    buffer<range<2>, 1> BufRange(&Range2, 1);
    buffer<int, 1> BufCounter(&Counter, 1);
    queue myQueue;

    myQueue.submit([&](handler &cgh) {
      auto AccRange = BufRange.get_access<access::mode::read_write>(cgh);
      auto AccCounter = BufCounter.get_access<access::mode::atomic>(cgh);
      cgh.parallel_for<class PF_init_id2>(Size, [=](id<2> ID) {
        AccCounter[0].fetch_add(1);
        AccRange[0][0] = this_id<2>()[0];
      });
    });
    myQueue.wait();
  }
  check("Counter = ", Counter, size * size);
}

void try_id3(size_t size) {
  range<3> Size{size, size, size};
  int Counter = 0;
  {
    buffer<range<3>, 1> BufRange(&Range3, 1);
    buffer<int, 1> BufCounter(&Counter, 1);
    queue myQueue;

    myQueue.submit([&](handler &cgh) {
      auto AccRange = BufRange.get_access<access::mode::read_write>(cgh);
      auto AccCounter = BufCounter.get_access<access::mode::atomic>(cgh);
      cgh.parallel_for<class PF_init_id3>(Size, [=](id<3> ID) {
        AccCounter[0].fetch_add(1);
        AccRange[0][0] = this_id<3>()[0];
      });
    });
    myQueue.wait();
  }
  check("Counter = ", Counter, size * size * size);
}

int main() {
  int x;

  x = 10;
  try_item1(x);
  try_item2(x);
  try_item3(x);
  try_id1(x);
  try_id2(x);
  try_id3(x);

  x = 1025;
  try_item1(x);
  try_item2(x);
  try_item3(x);
  try_id1(x);
  try_id2(x);
  try_id3(x);

  return 0;
}

// CHECK:  Size seen by user = 10
// CHECK-NEXT:  Counter = 10
// CHECK-NEXT:  Size seen by user = 10
// CHECK-NEXT:  Counter = 100
// CHECK-NEXT:  Size seen by user = 10
// CHECK-NEXT:  Counter = 1000
// CHECK-NEXT:  Counter = 10
// CHECK-NEXT:  Counter = 100
// CHECK-NEXT:  Counter = 1000
// CHECK-NEXT:  Size seen by user = 1025
// CHECK-NEXT:  Counter = 1025
// CHECK-NEXT:  Size seen by user = 1025
// CHECK-NEXT:  Counter = 1050625
// CHECK-NEXT:  Size seen by user = 1025
// CHECK-NEXT:  Counter = 1076890625
// CHECK-NEXT:  Counter = 1025
// CHECK-NEXT:  Counter = 1050625
// CHECK-NEXT:  Counter = 1076890625
