// RUN: %clangxx %s -o %t1.out -lsycl
// RUN: env SYCL_DEVICE_TYPE=HOST %t1.out
// RUN: %clangxx -fsycl %s -o %t2.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t2.out
// RUN: %CPU_RUN_PLACEHOLDER %t2.out
// RUN: %GPU_RUN_PLACEHOLDER %t2.out
// RUN: %ACC_RUN_PLACEHOLDER %t2.out
//==------------- buffer_full_copy.cpp - SYCL buffer basic test ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <cassert>

void check_copy_device_to_host(cl::sycl::queue &Queue) {
  const int size = 6, offset = 2;

  // Create 2d buffer
  cl::sycl::buffer<int, 2> simple_buffer(cl::sycl::range<2>(size, size));

  // Submit kernel where you'll fill the buffer
  Queue.submit([&](cl::sycl::handler &cgh) {
    auto acc = simple_buffer.get_access<cl::sycl::access::mode::write>(cgh);
    cgh.fill(acc, 13);
  });

  // Create host accessor with range and offset
  {
    cl::sycl::range<2> acc_range(2, 2);
    cl::sycl::id<2> acc_offset(offset, offset);
    auto acc = simple_buffer.get_access<cl::sycl::access::mode::read_write>(
        acc_range, acc_offset);
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j)
        acc[i][j] += 2;
    }
  }
  // Create host accessor with full access range
  // Check whether data was corrupted or not.
  {
    auto acc = simple_buffer.get_access<cl::sycl::access::mode::read>();
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j)
        if (offset <= i && i < offset + 2 && offset <= j && j < offset + 2) {
          assert(acc[i][j] == 15);
        } else {
          assert(acc[i][j] == 13);
        }
    }
  }
}

void check_fill(cl::sycl::queue &Queue) {
  const int size = 6, offset = 2;
  cl::sycl::buffer<float, 1> buf_1(size);
  cl::sycl::buffer<float, 1> buf_2(size / 2);
  std::vector<float> expected_res_1(size);
  std::vector<float> expected_res_2(size / 2);

  // fill with offset
  {
    auto acc = buf_1.get_access<cl::sycl::access::mode::write>();
    for (int i = 0; i < size; ++i) {
      acc[i] = i + 1;
      expected_res_1[i] = offset <= i && i < size / 2 + offset ? 1337.0 : i + 1;
    }
  }

  auto e = Queue.submit([&](cl::sycl::handler &cgh) {
    auto a = buf_1.template get_access<cl::sycl::access::mode::write>(
        cgh, size / 2, offset);
    cgh.fill(a, (float)1337.0);
  });
  e.wait();

  {
    auto acc_1 = buf_1.get_access<cl::sycl::access::mode::read>();
    for (int i = 0; i < size; ++i)
      assert(expected_res_1[i] == acc_1[i]);
  }
}

void check_copy_host_to_device(cl::sycl::queue &Queue) {
  const int size = 6, offset = 2;
  cl::sycl::buffer<float, 1> buf_1(size);
  cl::sycl::buffer<float, 1> buf_2(size / 2);
  std::vector<float> expected_res_1(size);
  std::vector<float> expected_res_2(size / 2);

  // copy acc 2 acc with offset
  {
    auto acc = buf_1.get_access<cl::sycl::access::mode::write>();
    for (int i = 0; i < size; ++i) {
      acc[i] = i + 1;
      expected_res_1[i] = i + 1;
    }
  }

  for (int i = 0; i < size / 2; ++i)
    expected_res_2[i] = expected_res_1[offset + i];

  auto e = Queue.submit([&](cl::sycl::handler &cgh) {
    auto a =
        buf_1.get_access<cl::sycl::access::mode::read>(cgh, size / 2, offset);
    auto b = buf_2.get_access<cl::sycl::access::mode::write>(cgh, size / 2);
    cgh.copy(a, b);
  });
  e.wait();

  {
    auto acc_1 = buf_1.get_access<cl::sycl::access::mode::read>();
    auto acc_2 = buf_2.get_access<cl::sycl::access::mode::read>();

    // check that there was no data corruption/loss
    for (int i = 0; i < size; ++i)
      assert(expected_res_1[i] == acc_1[i]);

    for (int i = 0; i < size / 2; ++i)
      assert(expected_res_2[i] == acc_2[i]);
  }

  cl::sycl::buffer<float, 2> buf_3({size, size});
  cl::sycl::buffer<float, 2> buf_4({size / 2, size / 2});
  std::vector<float> expected_res_3(size * size);
  std::vector<float> expected_res_4(size * size / 4);

  // copy acc 2 acc with offset for 2D buffers
  {
    auto acc = buf_3.get_access<cl::sycl::access::mode::write>();
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        acc[i][j] = i * size + j + 1;
        expected_res_3[i * size + j] = i * size + j + 1;
      }
    }
  }

  for (int i = 0; i < size / 2; ++i)
    for (int j = 0; j < size / 2; ++j)
      expected_res_4[i * size / 2 + j] =
          expected_res_3[(i + offset) * size + j + offset];

  e = Queue.submit([&](cl::sycl::handler &cgh) {
    auto a = buf_3.get_access<cl::sycl::access::mode::read>(
        cgh, {size / 2, size / 2}, {offset, offset});
    auto b = buf_4.get_access<cl::sycl::access::mode::write>(
        cgh, {size / 2, size / 2});
    cgh.copy(a, b);
  });
  e.wait();

  {
    auto acc_1 = buf_3.get_access<cl::sycl::access::mode::read>();
    auto acc_2 = buf_4.get_access<cl::sycl::access::mode::read>();

    // check that there was no data corruption/loss
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j)
        assert(expected_res_3[i * size + j] == acc_1[i][j]);
    }

    for (int i = 0; i < size / 2; ++i)
      for (int j = 0; j < size / 2; ++j)
        assert(expected_res_4[i * size / 2 + j] == acc_2[i][j]);
  }
}

int main() {
  try {
    cl::sycl::queue Queue;
    check_copy_host_to_device(Queue);
    check_copy_device_to_host(Queue);
    check_fill(Queue);
  } catch (cl::sycl::exception &ex) {
    std::cerr << ex.what() << std::endl;
  }

  return 0;
}
