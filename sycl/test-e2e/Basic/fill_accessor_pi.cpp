// RUN: %{build} -o %t.out
// RUN: %{run} %t.out | FileCheck %s

// the fill_accessor.cpp test checks that the .fill() function is working
// correctly. here we check that it uses the correct PI call.

#include <sycl/sycl.hpp>
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
    auto e1 = q.submit([&](sycl::handler &cgh) {
      auto acc1D = buffer_1D.get_access<sycl::access::mode::write>(cgh);
      cgh.fill(acc1D, float{1});
    });
    q.wait();

    std::cout << "start testFill_Buffer1D -- OFFSET" << std::endl;
    auto e1 = q.submit([&](sycl::handler &cgh) {
      auto acc1DOffset =
          buffer_1D.get_access<sycl::access::mode::write>(cgh, {2}, {4});
      cgh.fill(acc1DOffset, float{2});
    });
    q.wait();
  } // ~buffer
  assert(data_1D[1] == 1);
  assert(data_1D[2] == 2);
}

void testFill_Buffer2D() {
  std::cout << "start testFill_Buffer2D" << std::endl;
  std::vector<float> data_2D(total_2D, 0);
  {
    sycl::buffer<float, 2> buffer_2D(data_2D.data(),
                                     sycl::range<2>(height, width));

    sycl::queue q;
    q.submit([&](sycl::handler &cgh) {
      // cgh.depends_on(e1);
      // auto acc2D = buffer_2D.get_access<sycl::access::mode::write>(cgh); //
      // working now.

      // should go to parall_for, not MemoryManager::fill()
      auto acc2D =
          buffer_2D.get_access<sycl::access::mode::write>(cgh, {12, 8}, {4, 2});
      sycl::range<2> r = acc2D.get_range();
      std::cout << r[0] << "/" << r[1] << std::endl; // 32/64
      sycl::id<2> id = acc2D.get_offset();           // 0/0
      std::cout << "offset: " << id[0] << "/" << id[1] << std::endl;

      cgh.fill(acc2D, float{2});
    });

  } // ~buffer
  printData("Buffer2d", data_2D.data(), height * num);
  std::cout << "end testFill Buffer2d" << std::endl;
}