// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//==---------- reinterpret.cpp --- SYCL buffer reinterpret basic test ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

#include <climits>

// This tests verifies basic cases of using cl::sycl::buffer::reinterpret
// functionality - changing buffer type and range. This test checks that
// original buffer updates when we write to reinterpreted buffer and also checks
// that we can't create reinterpreted buffer when total size in bytes will be
// not same as total size in bytes of original buffer.

int main() {

  bool failed = false;
  cl::sycl::queue q;

  cl::sycl::range<1> r1(1);
  cl::sycl::range<1> r2(sizeof(unsigned int) / sizeof(unsigned char));
  cl::sycl::buffer<unsigned int, 1> buf_i(r1);
  auto buf_char = buf_i.reinterpret<unsigned char>(r2);
  q.submit([&](cl::sycl::handler &cgh) {
    auto acc = buf_char.get_access<cl::sycl::access::mode::read_write>(cgh);
    cgh.parallel_for<class chars>(
        r2, [=](cl::sycl::id<1> i) { acc[i] = UCHAR_MAX; });
  });

  {
    auto acc = buf_i.get_access<cl::sycl::access::mode::read>();
    if (acc[0] != UINT_MAX) {
      std::cout << acc[0] << std::endl;
      std::cout << "line: " << __LINE__ << " array[" << 0 << "] is " << acc[0]
                << " expected " << UINT_MAX << std::endl;
      failed = true;
    }
  }

  cl::sycl::range<1> r1d(9);
  cl::sycl::range<2> r2d(3, 3);
  cl::sycl::buffer<unsigned int, 1> buf_1d(r1d);
  auto buf_2d = buf_1d.reinterpret<unsigned int>(r2d);
  q.submit([&](cl::sycl::handler &cgh) {
    auto acc2d = buf_2d.get_access<cl::sycl::access::mode::read_write>(cgh);
    cgh.parallel_for<class ones>(r2d, [=](cl::sycl::item<2> itemID) {
      size_t i = itemID.get_id(0);
      size_t j = itemID.get_id(1);
      if (i == j)
        acc2d[i][j] = 1;
      else
        acc2d[i][j] = 0;
    });
  });

  {
    auto acc = buf_1d.get_access<cl::sycl::access::mode::read>();
    for (auto i = 0u; i < r1d.size(); i++) {
      size_t expected = (i % 4) ? 0 : 1;
      if (acc[i] != expected) {
        std::cout << "line: " << __LINE__ << " array[" << i << "] is " << acc[i]
                  << " expected " << expected << std::endl;
        failed = true;
      }
    }
  }

  try {
    cl::sycl::buffer<float, 1> buf_fl(r1d);
    auto buf_d = buf_1d.reinterpret<double>(r2d);
  } catch (cl::sycl::invalid_object_error e) {
    std::cout << "Expected exception has been caught: " << e.what()
              << std::endl;
  }

  // subbuffer reinterpret
  // 1d int -> char
  {
    std::size_t size = 12, offset = 4;
    std::vector<int> data(size + offset, 8);
    std::vector<int> expected_data(size + offset, 8);
    char *ptr = reinterpret_cast<char *>(&expected_data[offset]);
    for (int i = 0; i < size * sizeof(int); ++i) {
      *(ptr + i) = 13;
    }
    {
      cl::sycl::range<1> rng(size + offset);
      cl::sycl::buffer<int, 1> buffer_1(data.data(), rng);
      cl::sycl::buffer<int, 1> subbuffer_1(buffer_1, cl::sycl::id<1>(offset),
                                           cl::sycl::range<1>(size));
      cl::sycl::buffer<char, 1> reinterpret_subbuffer(
          subbuffer_1.reinterpret<char, 1>(
              cl::sycl::range<1>(subbuffer_1.get_size())));

      cl::sycl::queue cmd_queue;

      cmd_queue.submit([&](cl::sycl::handler &cgh) {
        auto rb_acc = reinterpret_subbuffer
                          .get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.parallel_for<class foo_1>(
            cl::sycl::range<1>(reinterpret_subbuffer.get_count()),
            [=](cl::sycl::id<1> index) { rb_acc[index] = 13; });
      });
    }

    for (std::size_t i = 0; i < size + offset; ++i) {
      assert(data[i] == expected_data[i]);
    }
  }

  // 1d char -> int
  {
    std::size_t size = 12, offset = 4;
    std::vector<char> data(size + offset, 8);
    std::vector<char> expected_data(size + offset, 8);
    for (std::size_t i = offset; i < size + offset; ++i) {
      expected_data[i] = i % sizeof(int) == 0 ? 1 : 0;
    }

    {
      cl::sycl::range<1> rng(size + offset);
      cl::sycl::buffer<char, 1> buffer_1(data.data(), rng);
      cl::sycl::buffer<char, 1> subbuffer_1(buffer_1, cl::sycl::id<1>(offset),
                                            cl::sycl::range<1>(size));
      cl::sycl::buffer<int, 1> reinterpret_subbuffer =
          subbuffer_1.reinterpret<int, 1>(
              cl::sycl::range<1>(subbuffer_1.get_size() / sizeof(int)));

      cl::sycl::queue cmd_queue;
      cmd_queue.submit([&](cl::sycl::handler &cgh) {
        auto rb_acc = reinterpret_subbuffer
                          .get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.parallel_for<class foo_2>(
            cl::sycl::range<1>(reinterpret_subbuffer.get_count()),
            [=](cl::sycl::id<1> index) { rb_acc[index] = 1; });
      });
    }

    for (std::size_t i = 0; i < size + offset; ++i) {
      assert(data[i] == expected_data[i]);
    }
  }

  // reinterpret 2D buffer to 1D buffer (same data type)
  // create subbuffer from 1D buffer with an offset
  // reinterpret subbuffer as 1D buffer of different data type
  {
    std::size_t size = 4, offset = 2, total_size = size + offset;
    cl::sycl::range<2> rng(total_size, total_size);

    std::vector<int> data(total_size * total_size, 8);
    std::vector<int> expected_data(total_size * total_size, 8);
    std::fill(expected_data.begin() + offset, expected_data.end(), 8);
    char *ptr =
        reinterpret_cast<char *>(&expected_data[offset * total_size + offset]);
    for (int i = 0; i < size * sizeof(int); ++i) {
      *(ptr + i) = 13;
    }

    {
      cl::sycl::buffer<int, 2> buffer_2d(data.data(), rng);
      cl::sycl::buffer<int, 1> buffer_1d = buffer_2d.reinterpret<int, 1>(
          cl::sycl::range<1>(buffer_2d.get_count()));
      // let's make an offset like for 2d buffer {offset, offset}
      // with a range = size elements
      cl::sycl::buffer<int, 1> subbuffer_1d(
          buffer_1d, cl::sycl::id<1>(offset * total_size + offset),
          cl::sycl::range<1>(size));

      cl::sycl::buffer<char, 1> reinterpret_subbuf =
          subbuffer_1d.reinterpret<char, 1>(subbuffer_1d.get_size());

      cl::sycl::queue cmd_queue;
      cmd_queue.submit([&](cl::sycl::handler &cgh) {
        auto rb_acc =
            reinterpret_subbuf.get_access<cl::sycl::access::mode::write>(cgh);
        cgh.parallel_for<class foo_3>(
            reinterpret_subbuf.get_range(),
            [=](cl::sycl::id<1> index) { rb_acc[index] = 13; });
      });
    }

    for (std::size_t i = 0; i < total_size; ++i)
      for (std::size_t j = 0; j < total_size; ++j)
        assert(data[i * total_size + j] == expected_data[i * total_size + j]);
  }
  return failed;
}
