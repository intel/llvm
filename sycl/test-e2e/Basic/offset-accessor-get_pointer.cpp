// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Per the SYCL 2020 spec (4.7.6.12 and others)
// accessor::get_pointer() returns a pointer to the start of this accessor’s
// memory. For a buffer accessor this is a pointer to the start of the
// underlying buffer, even if this is a ranged accessor whose range does not
// start at the beginning of the buffer.

// This is a departure from how get_pointer() was interpreted with offset
// accessors in the past. Not relevant for images, which do not support offset
// accessors.

#include <sycl/detail/core.hpp>
#include <vector>
using namespace sycl;

void test_across_ranges() {
  constexpr auto r_w = access::mode::read_write;
  constexpr unsigned long width = 4;
  constexpr unsigned long count = width * width;
  constexpr unsigned long count3D = width * width * width; // 64
  std::vector<int> v1(count);                              // for 1D testing.
  std::vector<int> v2(count);                              // for 2D testing.
  std::vector<int> v3(count3D);                            // 3D

  range<1> range_1D(count);
  range<2> range_2D(width, width);
  range<3> range_3D(width, width, width);

  queue myQueue;
  {
    // 1D, 2D, 3D
    buffer<int> buf_1D(v1.data(), count);
    buffer<int, 2> buf_2D(v2.data(), range_2D);
    buffer<int, 3> buf_3D(v3.data(), range_3D);

    myQueue.submit([&](handler &cgh) {
      auto acc_1D = buf_1D.get_access<r_w>(cgh, {2}, {10});
      auto acc_2D = buf_2D.get_access<r_w>(cgh, {2, 2}, {1, 1});
      auto acc_3D = buf_3D.get_access<r_w>(cgh, {2, 2, 2}, {1, 1, 1});
      cgh.single_task<class task>([=] {
        acc_1D.get_pointer()[0] = 5; // s.b. offset 0
        acc_1D[0] = 15;              // s.b. offset 10

        // 2D
        acc_2D.get_pointer()[0] = 7; // s.b. offset 0
        acc_2D[{0, 0}] = 17;         // s.b. offset {1,1} aka 5 if linear.

        // 3D
        acc_3D.get_pointer()[0] = 9; // s.b. offset 0
        acc_3D[{0, 0, 0}] = 19;      // s.b. offset {1,1,1} aka 21 if linear.
      });
    });
    myQueue.wait();
    // now host access - we offset by one more than the device test
    auto acc_1D = buf_1D.get_access<r_w>({2}, {11});
    auto acc_2D = buf_2D.get_access<r_w>({2, 2}, {1, 2});
    auto acc_3D = buf_3D.get_access<r_w>({2, 2, 2}, {1, 1, 2});
    acc_1D.get_pointer()[1] = 4; // s.b. offset 1
    acc_1D[0] = 14;              // s.b. offset 11

    // 2D
    acc_2D.get_pointer()[1] = 6; // s.b. offset 1
    acc_2D[{0, 0}] = 16;         // s.b. offset {1,2} aka 6 if linear.

    // 3D
    acc_3D.get_pointer()[1] = 8; // s.b. offset 1
    acc_3D[{0, 0, 0}] = 18;      // s.b. offset {1,1,2} aka 22 if linear.
  }                              //~buffer
  // always nice to have some feedback
  std::cout << "DEVICE" << std::endl;
  std::cout << "1D CHECK:  v1[0] should be 5: " << v1[0]
            << ", and v1[10] s.b. 15: " << v1[10] << std::endl;
  std::cout << "2D CHECK:  v2[0] should be 7: " << v2[0]
            << ", and v2[5]  s.b. 17: " << v2[5] << std::endl;
  std::cout << "3D CHECK:  v3[0] should be 9: " << v3[0]
            << ", and v3[21] s.b. 19: " << v3[21] << std::endl
            << std::endl;

  std::cout << "HOST" << std::endl;
  std::cout << "1D CHECK:  v1[1] should be 4: " << v1[1]
            << ", and v1[11] s.b. 14: " << v1[11] << std::endl;
  std::cout << "2D CHECK:  v2[1] should be 6: " << v2[1]
            << ", and v2[6]  s.b. 16: " << v2[6] << std::endl;
  std::cout << "3D CHECK:  v3[1] should be 8: " << v3[1]
            << ", and v3[22] s.b. 17: " << v3[22] << std::endl
            << std::endl;

  // device
  assert(v1[0] == 5);
  assert(v1[10] == 15);
  assert(v2[0] == 7);
  assert(v2[5] == 17); // offset {1,1} in a 4x4 field is linear offset 5
  assert(v3[0] == 9);
  assert(v3[21] == 19); // offset {1,1,1} in a 4x4x4 field is linear offset 21

  // host
  assert(v1[1] == 4);
  assert(v1[11] == 14);
  assert(v2[1] == 6);
  assert(v2[6] == 16); // offset {1,2} in a 4x4 field is linear offset 6
  assert(v3[1] == 8);
  assert(v3[22] == 18); // offset {1,1,2} in a 4x4x4 field is linear offset 22
}

int main() {
  test_across_ranges();

  std::cout << "OK!" << std::endl;

  return 0;
}
