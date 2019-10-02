// RUN: %clangxx -fsycl %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//==---------- subbuffer.cpp --- sub-buffer basic test ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Checks for:
// 1) Correct results after usage of different type of accessors to sub buffer
// 2) Exceptions if we trying to create sub buffer not according to spec

#include <CL/sycl.hpp>
#include <iostream>
#include <numeric>
#include <vector>

void checkHostAccessor(cl::sycl::queue &q) {
  std::size_t size =
      q.get_device().get_info<cl::sycl::info::device::mem_base_addr_align>() /
      8;
  size /= sizeof(int);
  size *= 2;
  std::vector<int> data(size);
  std::iota(data.begin(), data.end(), 0);
  {
    cl::sycl::buffer<int, 1> buf(data.data(), size);
    cl::sycl::buffer<int, 1> subbuf(buf, {size / 2}, {10});

    {
      auto host_acc = subbuf.get_access<cl::sycl::access::mode::write>();
      for (int i = 0; i < 10; ++i)
        host_acc[i] *= 10;
    }

    q.submit([&](cl::sycl::handler &cgh) {
      auto acc = subbuf.get_access<cl::sycl::access::mode::write>(cgh);
      cgh.parallel_for<class foobar_2>(
          cl::sycl::range<1>(10), [=](cl::sycl::id<1> i) { acc[i] *= -10; });
    });

    {
      auto host_acc = subbuf.get_access<cl::sycl::access::mode::read>();
      for (int i = 0; i < 10; ++i)
        assert(host_acc[i] == ((size / 2 + i) * -100) &&
               "Sub buffer host accessor test failed.");
    }
  }
  assert(data[0] == 0 && data[size - 1] == (size - 1) &&
         data[size / 2] == (size / 2 * -100) && "Loss of data");
}

void check1DSubBuffer(cl::sycl::queue &q) {
  std::size_t size =
      q.get_device().get_info<cl::sycl::info::device::mem_base_addr_align>() /
      8;
  size /= sizeof(int);
  size *= 2;

  std::size_t offset = size / 2, subbuf_size = 10, offset_inside_subbuf = 3,
              subbuffer_access_range = 10;
  std::vector<int> vec(size);
  std::vector<int> vec2(subbuf_size, 0);
  std::iota(vec.begin(), vec.end(), 0);

  try {
    cl::sycl::buffer<int, 1> buf(vec.data(), size);
    cl::sycl::buffer<int, 1> buf2(vec2.data(), subbuf_size);
    cl::sycl::buffer<int, 1> subbuf(buf, cl::sycl::id<1>(offset),
                                    cl::sycl::range<1>(subbuf_size));

    q.submit([&](cl::sycl::handler &cgh) {
      auto acc = subbuf.get_access<cl::sycl::access::mode::read_write>(
          cgh, cl::sycl::range<1>(subbuffer_access_range),
          cl::sycl::id<1>(offset_inside_subbuf));
      cgh.parallel_for<class foobar>(
          cl::sycl::range<1>(subbuffer_access_range - offset_inside_subbuf),
          [=](cl::sycl::id<1> i) { acc[i] *= -1; });
    });

    q.submit([&](cl::sycl::handler &cgh) {
      auto acc_sub = subbuf.get_access<cl::sycl::access::mode::read>(cgh);
      auto acc_buf = buf2.get_access<cl::sycl::access::mode::write>(cgh);
      cgh.parallel_for<class foobar_0>(
          subbuf.get_range(),
          [=](cl::sycl::id<1> i) { acc_buf[i] = acc_sub[i]; });
    });

    q.submit([&](cl::sycl::handler &cgh) {
      auto acc_sub = subbuf.get_access<cl::sycl::access::mode::read_write>(
          cgh, cl::sycl::range<1>(subbuffer_access_range));
      cgh.parallel_for<class foobar_1>(
          cl::sycl::range<1>(subbuffer_access_range),
          [=](cl::sycl::id<1> i) { acc_sub[i] *= 10; });
    });
    q.wait_and_throw();

  } catch (const cl::sycl::exception &e) {
    std::cerr << e.what() << std::endl;
    assert(false && "Exception was caught");
  }

  for (int i = offset; i < subbuf_size; ++i)
    assert(vec[i] == (i > 34 ? i * 10 : i * -10) &&
           "Invalid result in 1d sub buffer");

  for (int i = 0; i < subbuf_size; ++i)
    assert(vec2[i] == (i < 3 ? (32 + i) : (32 + i) * -1) &&
           "Invalid result in 1d sub buffer");
}

void checkExceptions() {
  size_t row = 8, col = 8;
  std::vector<int> vec(1, 0);
  cl::sycl::buffer<int, 2> buf2d(vec.data(), {row, col});
  cl::sycl::buffer<int, 3> buf3d(vec.data(), {row / 2, col / 2, col / 2});

  // non-contiguous region
  try {
    cl::sycl::buffer<int, 2> sub_buf{buf2d, /*offset*/ cl::sycl::range<2>{2, 0},
                                     /*size*/ cl::sycl::range<2>{2, 2}};
    assert(!"non contiguous region exception wasn't caught");
  } catch (const cl::sycl::invalid_object_error &e) {
    std::cerr << e.what() << std::endl;
  }

  try {
    cl::sycl::buffer<int, 2> sub_buf{buf2d, /*offset*/ cl::sycl::range<2>{2, 2},
                                     /*size*/ cl::sycl::range<2>{2, 6}};
    assert(!"non contiguous region exception wasn't caught");
  } catch (const cl::sycl::invalid_object_error &e) {
    std::cerr << e.what() << std::endl;
  }

  try {
    cl::sycl::buffer<int, 3> sub_buf{buf3d,
                                     /*offset*/ cl::sycl::range<3>{0, 2, 1},
                                     /*size*/ cl::sycl::range<3>{1, 2, 3}};
    assert(!"non contiguous region exception wasn't caught");
  } catch (const cl::sycl::invalid_object_error &e) {
    std::cerr << e.what() << std::endl;
  }

  try {
    cl::sycl::buffer<int, 3> sub_buf{buf3d,
                                     /*offset*/ cl::sycl::range<3>{0, 0, 0},
                                     /*size*/ cl::sycl::range<3>{2, 3, 4}};
    assert(!"non contiguous region exception wasn't caught");
  } catch (const cl::sycl::invalid_object_error &e) {
    std::cerr << e.what() << std::endl;
  }

  // out of bounds
  try {
    cl::sycl::buffer<int, 2> sub_buf{buf2d, /*offset*/ cl::sycl::range<2>{2, 2},
                                     /*size*/ cl::sycl::range<2>{2, 8}};
    assert(!"out of bounds exception wasn't caught");
  } catch (const cl::sycl::invalid_object_error &e) {
    std::cerr << e.what() << std::endl;
  }

  try {
    cl::sycl::buffer<int, 3> sub_buf{buf3d,
                                     /*offset*/ cl::sycl::range<3>{1, 1, 1},
                                     /*size*/ cl::sycl::range<3>{1, 1, 4}};
    assert(!"out of bounds exception wasn't caught");
  } catch (const cl::sycl::invalid_object_error &e) {
    std::cerr << e.what() << std::endl;
  }

  try {
    cl::sycl::buffer<int, 3> sub_buf{buf3d,
                                     /*offset*/ cl::sycl::range<3>{3, 3, 0},
                                     /*size*/ cl::sycl::range<3>{1, 2, 4}};
    assert(!"out of bounds exception wasn't caught");
  } catch (const cl::sycl::invalid_object_error &e) {
    std::cerr << e.what() << std::endl;
  }

  // subbuffer from subbuffer
  try {
    cl::sycl::buffer<int, 2> sub_buf{buf2d, /*offset*/ cl::sycl::range<2>{2, 0},
                                     /*size*/ cl::sycl::range<2>{2, 8}};
    cl::sycl::buffer<int, 2> sub_sub_buf(sub_buf, cl::sycl::range<2>{0, 0},
                                         /*size*/ cl::sycl::range<2>{0, 0});
    assert(!"invalid subbuffer exception wasn't caught");
  } catch (const cl::sycl::invalid_object_error &e) {
    std::cerr << e.what() << std::endl;
  }
}

int main() {
  cl::sycl::queue q;
  check1DSubBuffer(q);
  checkHostAccessor(q);
  // TODO! Uncomment once SYCL-CTS are fixed
  // checkExceptions();
  return 0;
}
