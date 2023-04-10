// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple  %s -o %t2.out
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

#include <cassert>
#include <iostream>
#include <sycl/sycl.hpp>

void check_copy_device_to_host(sycl::queue &Queue) {
  constexpr int size = 6, offset = 2;

  // Create 2d buffer
  sycl::buffer<int, 2> simple_buffer(sycl::range<2>(size, size));

  // Submit kernel where you'll fill the buffer
  Queue.submit([&](sycl::handler &cgh) {
    auto acc = simple_buffer.get_access<sycl::access::mode::write>(cgh);
    cgh.fill(acc, 13);
  });

  // Create host accessor with range and offset
  {
    sycl::range<2> acc_range(2, 2);
    sycl::id<2> acc_offset(offset, offset);
    auto acc = simple_buffer.get_access<sycl::access::mode::read_write>(
        acc_range, acc_offset);
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j)
        acc[i][j] += 2;
    }
  }
  // Create host accessor with full access range
  // Check whether data was corrupted or not.
  {
    auto acc = simple_buffer.get_access<sycl::access::mode::read>();
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

void check_fill(sycl::queue &Queue) {
  constexpr int size = 6, offset = 2;
  sycl::buffer<float, 1> buf_1(size);
  sycl::buffer<float, 1> buf_2(size / 2);
  std::array<float, size> expected_res_1;
  std::array<float, size / 2> expected_res_2;

  // fill with offset
  {
    auto acc = buf_1.get_access<sycl::access::mode::write>();
    for (int i = 0; i < size; ++i) {
      acc[i] = i + 1;
      expected_res_1[i] = offset <= i && i < size / 2 + offset ? 1337.0 : i + 1;
    }
  }

  auto e = Queue.submit([&](sycl::handler &cgh) {
    auto a = buf_1.template get_access<sycl::access::mode::write>(cgh, size / 2,
                                                                  offset);
    cgh.fill(a, (float)1337.0);
  });
  e.wait();

  {
    auto acc_1 = buf_1.get_access<sycl::access::mode::read>();
    for (int i = 0; i < size; ++i)
      assert(expected_res_1[i] == acc_1[i]);
  }
}

void check_copy_host_to_device(sycl::queue &Queue) {
  constexpr int size = 6, offset = 2;
  sycl::buffer<float, 1> buf_1(size);
  sycl::buffer<float, 1> buf_2(size / 2);
  std::array<float, size> expected_res_1;
  std::array<float, size / 2> expected_res_2;

  // copy acc 2 acc with offset
  {
    auto acc = buf_1.get_access<sycl::access::mode::write>();
    for (int i = 0; i < size; ++i) {
      acc[i] = i + 1;
      expected_res_1[i] = i + 1;
    }
  }

  for (int i = 0; i < size / 2; ++i)
    expected_res_2[i] = expected_res_1[offset + i];

  auto e = Queue.submit([&](sycl::handler &cgh) {
    auto a = buf_1.get_access<sycl::access::mode::read>(cgh, size / 2, offset);
    auto b = buf_2.get_access<sycl::access::mode::write>(cgh, size / 2);
    cgh.copy(a, b);
  });
  e.wait();

  {
    auto acc_1 = buf_1.get_access<sycl::access::mode::read>();
    auto acc_2 = buf_2.get_access<sycl::access::mode::read>();

    // check that there was no data corruption/loss
    for (int i = 0; i < size; ++i)
      assert(expected_res_1[i] == acc_1[i]);

    for (int i = 0; i < size / 2; ++i)
      assert(expected_res_2[i] == acc_2[i]);
  }

  sycl::buffer<float, 2> buf_3({size, size});
  sycl::buffer<float, 2> buf_4({size / 2, size / 2});
  std::array<std::array<float, size>, size> expected_res_3;
  std::array<std::array<float, size / 2>, size / 2> expected_res_4;

  // copy acc 2 acc with offset for 2D buffers
  {
    auto acc = buf_3.get_access<sycl::access::mode::write>();
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        acc[i][j] = i * size + j + 1;
        expected_res_3[i][j] = acc[i][j];
      }
    }
  }

  for (int i = 0; i < size / 2; ++i)
    for (int j = 0; j < size / 2; ++j)
      expected_res_4[i][j] = expected_res_3[i + offset][j + offset];

  e = Queue.submit([&](sycl::handler &cgh) {
    auto a = buf_3.get_access<sycl::access::mode::read>(
        cgh, {size / 2, size / 2}, {offset, offset});
    auto b =
        buf_4.get_access<sycl::access::mode::write>(cgh, {size / 2, size / 2});
    cgh.copy(a, b);
  });
  e.wait();

  {
    auto acc_1 = buf_3.get_access<sycl::access::mode::read>();
    auto acc_2 = buf_4.get_access<sycl::access::mode::read>();

    // check that there was no data corruption/loss
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j)
        assert(expected_res_3[i][j] == acc_1[i][j]);
    }

    for (int i = 0; i < size / 2; ++i)
      for (int j = 0; j < size / 2; ++j)
        assert(expected_res_4[i][j] == acc_2[i][j]);
  }

  sycl::buffer<float, 3> buf_5({size, size, size});
  sycl::buffer<float, 3> buf_6({size / 2, size / 2, size / 2});
  std::array<std::array<std::array<float, size>, size>, size> expected_res_5;
  std::array<std::array<std::array<float, size / 2>, size / 2>, size / 2>
      expected_res_6;

  // copy acc 2 acc with offset for 3D buffers
  {
    auto acc = buf_5.get_access<sycl::access::mode::write>();
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        for (int k = 0; k < size; ++k) {
          acc[i][j][k] = (i * size * size) + (j * size) + k + 1;
          expected_res_5[i][j][k] = (i * size * size) + (j * size) + k + 1;
        }
      }
    }
  }

  for (int i = 0; i < size / 2; ++i)
    for (int j = 0; j < size / 2; ++j)
      for (int k = 0; k < size / 2; ++k)
        expected_res_6[i][j][k] =
            expected_res_5[i + offset][j + offset][k + offset];

  e = Queue.submit([&](sycl::handler &cgh) {
    auto a = buf_5.get_access<sycl::access::mode::read>(
        cgh, {size / 2, size / 2, size / 2}, {offset, offset, offset});
    auto b = buf_6.get_access<sycl::access::mode::write>(
        cgh, {size / 2, size / 2, size / 2});
    cgh.copy(a, b);
  });
  e.wait();

  {
    auto acc_1 = buf_5.get_access<sycl::access::mode::read>();
    auto acc_2 = buf_6.get_access<sycl::access::mode::read>();

    // check that there was no data corruption/loss
    for (int i = 0; i < size; ++i)
      for (int j = 0; j < size; ++j)
        for (int k = 0; k < size; ++k)
          assert(expected_res_5[i][j][k] == acc_1[i][j][k]);

    for (int i = 0; i < size / 2; ++i) {
      for (int j = 0; j < size / 2; ++j)
        for (int k = 0; k < size / 2; ++k)
          assert(expected_res_6[i][j][k] == acc_2[i][j][k]);
    }
  }
}

void check_exception_code() {
  sycl::queue q;

  const size_t bufSize = 10;
  std::vector<int> res(bufSize);
  // std::iota(res.begin(), res.end(), 1);
  sycl::range<1> r(bufSize);
  sycl::buffer<int, 1> b(res.data(), r);
  sycl::range<1> smallRange(bufSize / 2);
  sycl::id<1> offset(bufSize);

  try {
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor src(b, cgh);
      sycl::accessor destToSmall(b, cgh, smallRange);
      cgh.copy(src, destToSmall);
    });
    q.wait_and_throw();

    assert(false &&
           "copy with too small Dest arg should have thrown an exception");
  } catch (sycl::exception e) {
    assert(e.code() == sycl::errc::invalid);
  }
}

int main() {
  try {
    sycl::queue Queue;
    check_copy_host_to_device(Queue);
    check_copy_device_to_host(Queue);
    check_fill(Queue);
    check_exception_code();
  } catch (sycl::exception &ex) {
    std::cerr << ex.what() << std::endl;
    return 1;
  }

  return 0;
}
