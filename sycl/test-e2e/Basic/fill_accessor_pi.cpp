// RUN: %{build} -o %t.out
// RUN: env SYCL_PI_TRACE=2 %{run} %t.out | FileCheck %s

// This test merely checks the use of the correct PI call. Its sister test
// fill_accessor.cpp thoroughly checks the workings of the .fill() call.

#include <sycl/detail/core.hpp>
constexpr int width = 32;
constexpr int height = 16;
constexpr int depth = 8;
constexpr int total_2D = width * height;
constexpr int total_3D = width * height * depth;

void testFill_Buffer1D() {
  std::vector<float> data_1D(width, 0);
  {
    sycl::buffer<float, 1> buffer_1D(data_1D.data(), sycl::range<1>(width));

    sycl::queue q;
    std::cout << "start testFill_Buffer1D" << std::endl;
    q.submit([&](sycl::handler &cgh) {
      auto acc1D = buffer_1D.get_access<sycl::access::mode::write>(cgh);
      // should stage piEnqueueMemBufferFill
      cgh.fill(acc1D, float{1});
    });
    q.wait();

    std::cout << "start testFill_Buffer1D -- OFFSET" << std::endl;
    q.submit([&](sycl::handler &cgh) {
      auto acc1DOffset =
          buffer_1D.get_access<sycl::access::mode::write>(cgh, {4}, {2});
      // despite being offset, should stage piEnqueueMemBufferFill
      cgh.fill(acc1DOffset, float{2});
    });
    q.wait();
  } // ~buffer

  // quick check.  fill_accessor.cpp is more thorough.
  assert(data_1D[1] == 1);
  assert(data_1D[2] == 2);
}

void testFill_Buffer2D() {
  std::vector<float> data_2D(total_2D, 0);
  {
    sycl::buffer<float, 2> buffer_2D(data_2D.data(),
                                     sycl::range<2>(height, width));

    sycl::queue q;
    std::cout << "start testFill_Buffer2D" << std::endl;
    q.submit([&](sycl::handler &cgh) {
      auto acc2D = buffer_2D.get_access<sycl::access::mode::write>(cgh);
      // should stage piEnqueueMemBufferFill
      cgh.fill(acc2D, float{3});
    });
    q.wait();

    std::cout << "start testFill_Buffer2D -- OFFSET" << std::endl;
    q.submit([&](sycl::handler &cgh) {
      auto acc2D =
          buffer_2D.get_access<sycl::access::mode::write>(cgh, {8, 12}, {2, 2});
      // "ranged accessor" will have to be handled by custom kernel:
      // piEnqueueKernelLaunch
      cgh.fill(acc2D, float{4});
    });
    q.wait();
  } // ~buffer

  // quick check.  fill_accessor.cpp is more thorough.
  assert(data_2D[(1 * width) + 1] == 3); // [1][1] sb 3
  assert(data_2D[(2 * width) + 2] == 4); // [2][2] sb 4
}

void testFill_Buffer3D() {
  std::vector<float> data_3D(total_3D, 0);
  {
    sycl::buffer<float, 3> buffer_3D(data_3D.data(),
                                     sycl::range<3>(depth, height, width));

    sycl::queue q;
    std::cout << "start testFill_Buffer3D" << std::endl;
    q.submit([&](sycl::handler &cgh) {
      auto acc3D = buffer_3D.get_access<sycl::access::mode::write>(cgh);
      // should stage piEnqueueMemBufferFill
      cgh.fill(acc3D, float{5});
    });
    q.wait();

    std::cout << "start testFill_Buffer3D -- OFFSET" << std::endl;
    q.submit([&](sycl::handler &cgh) {
      auto acc3D = buffer_3D.get_access<sycl::access::mode::write>(
          cgh, {4, 8, 12}, {3, 3, 3});
      // "ranged accessor" will have to be handled by custom kernel:
      // piEnqueueKernelLaunch
      cgh.fill(acc3D, float{6});
    });
    q.wait();
  } // ~buffer

  // quick check.  fill_accessor.cpp is more thorough.
  assert(data_3D[(1 * height * width) + (1 * width) + 1] ==
         5); // [1][1][1] sb 5
  assert(data_3D[(3 * height * width) + (3 * width) + 3] ==
         6); // [3][3][3] sb 6
}

void testFill_ZeroDim() {
  sycl::range<1> r{1};
  std::vector<float> data_0D(1, 0);
  {
    sycl::buffer<float, 1> Buffer(data_0D.data(), r);
    sycl::queue q;
    std::cout << "start testFill_ZeroDim" << std::endl;
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor<float, 0, sycl::access::mode::write> Acc0(Buffer, cgh);
      cgh.fill(Acc0, float{1});
    });
    q.wait();
  }
  assert(data_0D[0] == 1);
}

int main() {
  testFill_Buffer1D();
  testFill_Buffer2D();
  testFill_Buffer3D();
  testFill_ZeroDim();
  return 0;
}

// CHECK: start testFill_Buffer1D
// CHECK: piEnqueueMemBufferFill
// CHECK: start testFill_Buffer1D -- OFFSET
// CHECK: piEnqueueMemBufferFill

// CHECK: start testFill_Buffer2D
// CHECK: piEnqueueMemBufferFill
// CHECK: start testFill_Buffer2D -- OFFSET
// CHECK: piEnqueueKernelLaunch

// CHECK: start testFill_Buffer3D
// CHECK: piEnqueueMemBufferFill
// CHECK: start testFill_Buffer3D -- OFFSET
// CHECK: piEnqueueKernelLaunch

// CHECK: start testFill_ZeroDim
// CHECK: piEnqueueMemBufferFill
