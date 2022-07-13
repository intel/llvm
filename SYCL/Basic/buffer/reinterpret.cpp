// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple  %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//
// XFAIL: level_zero&&gpu

//==---------- reinterpret.cpp --- SYCL buffer reinterpret basic test ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <climits>

// This tests verifies basic cases of using cl::sycl::buffer::reinterpret
// functionality - changing buffer type and range. This test checks that
// original buffer updates when we write to reinterpreted buffer and also checks
// that we can't create reinterpreted buffer when total size in bytes will be
// not same as total size in bytes of original buffer.

template <class KernelName, typename NewType, typename OldType, int dim>
void execute_kernel(cl::sycl::queue &cmd_queue, std::vector<OldType> &data,
                    const cl::sycl::range<dim> &buffer_range,
                    const cl::sycl::id<dim> &sub_buf_offset,
                    const cl::sycl::range<dim> &sub_buf_range,
                    const NewType &val) {
  cl::sycl::buffer<OldType, dim> buffer(data.data(), buffer_range);
  cl::sycl::buffer<OldType, dim> subbuffer(buffer, sub_buf_offset,
                                           sub_buf_range);
  cl::sycl::range<dim> reinterpret_range = subbuffer.get_range();
  reinterpret_range[dim - 1] =
      reinterpret_range[dim - 1] * sizeof(OldType) / sizeof(NewType);
  cl::sycl::buffer<NewType, dim> reinterpret_subbuf =
      subbuffer.template reinterpret<NewType, dim>(reinterpret_range);

  cmd_queue.submit([&](cl::sycl::handler &cgh) {
    auto rb_acc =
        reinterpret_subbuf.template get_access<cl::sycl::access::mode::write>(
            cgh);
    cgh.parallel_for<KernelName>(
        reinterpret_subbuf.get_range(),
        [=](cl::sycl::id<dim> index) { rb_acc[index] = val; });
  });
  cmd_queue.wait_and_throw();
}

int main() {

  bool failed = false;
  cl::sycl::queue cmd_queue;
  size_t mem_base_align_in_bytes =
      cmd_queue.get_device()
          .get_info<cl::sycl::info::device::mem_base_addr_align>() /
      8;

  cl::sycl::range<1> r1(1);
  cl::sycl::range<1> r2(sizeof(unsigned int) / sizeof(unsigned char));
  cl::sycl::buffer<unsigned int, 1> buf_i(r1);
  auto buf_char = buf_i.reinterpret<unsigned char>(r2);
  cmd_queue.submit([&](cl::sycl::handler &cgh) {
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
  cmd_queue.submit([&](cl::sycl::handler &cgh) {
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
    std::size_t offset = mem_base_align_in_bytes / sizeof(int);
    std::size_t sub_buf_size = 16;
    std::vector<int> data(sub_buf_size + offset, 8);
    std::vector<int> expected_data(sub_buf_size + offset, 8);
    char *ptr = reinterpret_cast<char *>(&expected_data[offset]);
    char val = 13;
    for (int i = 0; i < sub_buf_size * sizeof(int); ++i) {
      *(ptr + i) = val;
    }

    execute_kernel<class foo_1>(
        cmd_queue, data, cl::sycl::range<1>{sub_buf_size + offset},
        cl::sycl::id<1>{offset}, cl::sycl::range<1>{sub_buf_size}, val);

    for (std::size_t i = 0; i < sub_buf_size + offset; ++i) {
      assert(data[i] == expected_data[i] &&
             "1D sub buffer int->char reinterpret failed");
    }
  }

  // 1d char -> int
  {
    std::size_t offset = mem_base_align_in_bytes / sizeof(char);
    std::size_t sub_buf_size = 16;

    std::vector<char> data(sub_buf_size + offset, 8);
    std::vector<char> expected_data(sub_buf_size + offset, 8);
    int val = 0xaabbccdd;
    int *ptr = reinterpret_cast<int *>(&expected_data[offset]);
    for (std::size_t i = 0; i < sub_buf_size / sizeof(int); ++i) {
      *(ptr + i) = val;
    }

    execute_kernel<class foo_2>(
        cmd_queue, data, cl::sycl::range<1>{sub_buf_size + offset},
        cl::sycl::id<1>{offset}, cl::sycl::range<1>{sub_buf_size}, val);

    for (std::size_t i = 0; i < sub_buf_size + offset; ++i) {
      assert(data[i] == expected_data[i] &&
             "1D sub buffer char->int reinterpret failed");
    }
  }

  // reinterpret 2D buffer to 1D buffer (same data type)
  // create subbuffer from 1D buffer with an offset
  // reinterpret subbuffer as 1D buffer of different data type
  {

    std::size_t cols = mem_base_align_in_bytes / sizeof(int);
    std::size_t rows = 6, offset_rows = 2;
    cl::sycl::range<2> rng(rows, cols);

    std::vector<int> data(rows * cols, 8);
    std::vector<int> expected_data(rows * cols, 8);
    char *ptr = reinterpret_cast<char *>(&expected_data[offset_rows * cols]);
    for (int i = 0; i < cols * sizeof(int); ++i) {
      *(ptr + i) = 13;
    }

    {
      cl::sycl::buffer<int, 2> buffer_2d(data.data(), rng);
      cl::sycl::buffer<int, 1> buffer_1d = buffer_2d.reinterpret<int, 1>(
          cl::sycl::range<1>(buffer_2d.get_count()));
      // let's make an offset like for 2d buffer {offset_rows, cols}
      // with a range = {1, cols}
      cl::sycl::buffer<int, 1> subbuffer_1d(buffer_1d,
                                            cl::sycl::id<1>(offset_rows * cols),
                                            cl::sycl::range<1>(cols));

      cl::sycl::buffer<char, 1> reinterpret_subbuf =
          subbuffer_1d.reinterpret<char, 1>(subbuffer_1d.get_size());

      cmd_queue.submit([&](cl::sycl::handler &cgh) {
        auto rb_acc =
            reinterpret_subbuf.get_access<cl::sycl::access::mode::write>(cgh);
        cgh.parallel_for<class foo_3>(
            reinterpret_subbuf.get_range(),
            [=](cl::sycl::id<1> index) { rb_acc[index] = 13; });
      });
      cmd_queue.wait_and_throw();
    }

    for (std::size_t i = 0; i < rows; ++i)
      for (std::size_t j = 0; j < cols; ++j)
        assert(data[i * cols + j] == expected_data[i * cols + j] &&
               "2D->1D->sub buffer reinterpret failed");
  }

  // 2D
  // Offset = {/*Rows*/ Any, /*Cols*/ 0},
  // Range  = {/*Rows*/ Any, /*Cols*/ <original_buffer>}
  // int -> char
  {
    std::size_t offset_rows = 1, offset_cols = 0, subbuffer_rows = 1,
                buf_row = 3, buf_col = mem_base_align_in_bytes / sizeof(int);
    std::vector<int> data(buf_col * buf_row, 8);
    std::vector<int> expected_data(buf_col * buf_row, 8);

    char val = 13;
    char *ptr = reinterpret_cast<char *>(
        &expected_data[offset_rows * buf_col + offset_cols]);
    for (int i = 0; i < buf_col * subbuffer_rows * sizeof(int); ++i) {
      *(ptr + i) = val;
    }

    execute_kernel<class foo_4>(
        cmd_queue, data, cl::sycl::range<2>{buf_row, buf_col},
        cl::sycl::id<2>{offset_rows, offset_cols},
        cl::sycl::range<2>{subbuffer_rows, buf_col}, val);

    for (std::size_t i = 0; i < buf_row; ++i)
      for (std::size_t j = 0; j < buf_col; ++j)
        assert(data[i * buf_col + j] == expected_data[i * buf_col + j] &&
               "2D sub buffer int->char reinterpret failed");
  }

  // char -> int
  {
    std::size_t offset_rows = 1, offset_cols = 0, subbuffer_rows = 1,
                buf_row = 3, buf_col = mem_base_align_in_bytes / sizeof(char);
    std::vector<char> data(buf_col * buf_row, 8);
    std::vector<char> expected_data(buf_col * buf_row, 8);

    int val = 0xaabbccdd;
    int *ptr = reinterpret_cast<int *>(
        &expected_data[offset_rows * buf_col + offset_cols]);
    for (int i = 0; i < buf_col * subbuffer_rows / sizeof(int); ++i) {
      *(ptr + i) = val;
    }

    execute_kernel<class foo_5>(
        cmd_queue, data, cl::sycl::range<2>{buf_row, buf_col},
        cl::sycl::id<2>{offset_rows, offset_cols},
        cl::sycl::range<2>{subbuffer_rows, buf_col}, val);

    for (std::size_t i = 0; i < buf_row; ++i)
      for (std::size_t j = 0; j < buf_col; ++j)
        assert(data[i * buf_col + j] == expected_data[i * buf_col + j] &&
               "2D sub buffer int->char reinterpret failed");
  }

  return failed;
}
