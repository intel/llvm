// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple  %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// UNSUPPORTED: (opencl && gpu)

//
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

#include <sycl/sycl.hpp>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

void checkHostAccessor(sycl::queue &q) {
  std::size_t subbuf_align =
      q.get_device().get_info<sycl::info::device::mem_base_addr_align>() / 8;
  std::size_t size =
      std::max(subbuf_align, 10 * 2 * sizeof(int)); // hold at least 20 elements
  size /= sizeof(int);
  size *= 2;

  std::vector<int> data(size);
  std::iota(data.begin(), data.end(), 0);
  {
    sycl::buffer<int, 1> buf(data.data(), size);
    sycl::buffer<int, 1> subbuf(buf, {size / 2}, {10});

    {
      auto host_acc = subbuf.get_access<sycl::access::mode::write>();
      for (int i = 0; i < 10; ++i)
        host_acc[i] *= 10;
    }

    q.submit([&](sycl::handler &cgh) {
      auto acc = subbuf.get_access<sycl::access::mode::write>(cgh);
      cgh.parallel_for<class foobar_2>(sycl::range<1>(10),
                                       [=](sycl::id<1> i) { acc[i] *= -10; });
    });

    {
      auto host_acc = subbuf.get_access<sycl::access::mode::read>();
      for (int i = 0; i < 10; ++i)
        assert(host_acc[i] == ((size / 2 + i) * -100) &&
               "Sub buffer host accessor test failed.");
    }
  }
  assert(data[0] == 0 && data[size - 1] == (size - 1) &&
         data[size / 2] == (size / 2 * -100) && "Loss of data");
}

void check1DSubBuffer(sycl::queue &q) {
  std::size_t subbuf_align =
      q.get_device().get_info<sycl::info::device::mem_base_addr_align>() / 8;
  std::size_t size =
      std::max(subbuf_align, 32 * sizeof(int)); // hold at least 32 elements
  size /= sizeof(int);
  size *= 2;

  std::size_t offset = size / 2, subbuf_size = 10, offset_inside_subbuf = 3,
              subbuffer_access_range = subbuf_size - offset_inside_subbuf; // 7.
  std::vector<int> vec(size);
  std::vector<int> vec2(subbuf_size, 0);
  std::iota(vec.begin(), vec.end(), 0);

  std::cout << "buffer size: " << size << ", subbuffer start: " << offset
            << std::endl;

  try {
    sycl::buffer<int, 1> buf(vec.data(), size);
    sycl::buffer<int, 1> buf2(vec2.data(), subbuf_size);
    // subbuffer is 10 elements, starting at midpoint. (typically 32)
    sycl::buffer<int, 1> subbuf(buf, sycl::id<1>(offset),
                                sycl::range<1>(subbuf_size));

    // test offset accessor against a subbuffer
    q.submit([&](sycl::handler &cgh) {
      // accessor starts at the third element of the subbuffer
      // and can read for 7 more (ie to the end of the subbuffer)
      auto acc = subbuf.get_access<sycl::access::mode::read_write>(
          cgh, sycl::range<1>(subbuffer_access_range),
          sycl::id<1>(offset_inside_subbuf));
      // subrange is made negative. ( 32 33 34 -35 -36 -37 -38 -39 -40 -41)
      cgh.parallel_for<class foobar>(sycl::range<1>(subbuffer_access_range),
                                     [=](sycl::id<1> i) { acc[i] *= -1; });
    });

    // copy results of last operation back to buf2/vec2
    q.submit([&](sycl::handler &cgh) {
      auto acc_sub = subbuf.get_access<sycl::access::mode::read>(cgh);
      auto acc_buf = buf2.get_access<sycl::access::mode::write>(cgh);
      cgh.parallel_for<class foobar_0>(
          subbuf.get_range(), [=](sycl::id<1> i) { acc_buf[i] = acc_sub[i]; });
    });

    // multiple entire subbuffer by 10.
    // now original buffer will be
    // (..29 30 31 | 320 330 340 -350 -360 -370 -380 -390 -400 -410 | 42 43 44
    // ...)
    q.submit([&](sycl::handler &cgh) {
      auto acc_sub = subbuf.get_access<sycl::access::mode::read_write>(
          cgh, sycl::range<1>(subbuf_size));
      cgh.parallel_for<class foobar_1>(
          sycl::range<1>(subbuf_size),
          [=](sycl::id<1> i) { acc_sub[i] *= 10; });
    });
    q.wait_and_throw();

    // buffers go out of scope. data must be copied back to vector no later than
    // this.
  } catch (const sycl::exception &e) {
    std::cerr << e.what() << std::endl;
    assert(false && "Exception was caught");
  }

  // check buffer data in the area of the subbuffer
  // OCL:GPU confused   =>  320 330 340 -350 -360 -370 -380 39   40   41
  // every other device =>  320 330 340 -350 -360 -370 -380 -390 -400 -410
  for (int i = offset; i < offset + subbuf_size; ++i)
    assert(vec[i] == (i < offset + offset_inside_subbuf ? i * 10 : i * -10) &&
           "Invalid result in buffer overlapped by 1d sub buffer");

  // check buffer data in the area OUTSIDE the subbuffer
  for (int i = 0; i < size; i++) {
    if (i < offset)
      assert(vec[i] == i && "data preceding subbuffer incorrectly altered");

    if (i > offset + subbuf_size)
      assert(vec[i] == i && "data following subbuffer incorrectly altered");
  }

  // check the copy of the subbuffer data after the first operation
  // OCL:GPU        => 32 33 34 -35 -36 -37 -38 0   0   0
  // everyone else  => 32 33 34 -35 -36 -37 -38 -39 -40 -41
  for (int i = 0; i < subbuf_size; ++i)
    assert(vec2[i] == (i < 3 ? (offset + i) : (offset + i) * -1) &&
           "Invalid result in captured 1d sub buffer, vec2");
}

void checkExceptions() {
  size_t row = 8, col = 8;
  std::vector<int> vec(1, 0);
  sycl::buffer<int, 2> buf2d(vec.data(), {row, col});
  sycl::buffer<int, 3> buf3d(vec.data(), {row / 2, col / 2, col / 2});

  // non-contiguous region
  try {
    sycl::buffer<int, 2> sub_buf{buf2d, /*offset*/ sycl::range<2>{2, 0},
                                 /*size*/ sycl::range<2>{2, 2}};
    assert(!"non contiguous region exception wasn't caught");
  } catch (const sycl::invalid_object_error &e) {
    std::cerr << e.what() << std::endl;
  }

  try {
    sycl::buffer<int, 2> sub_buf{buf2d, /*offset*/ sycl::range<2>{2, 2},
                                 /*size*/ sycl::range<2>{2, 6}};
    assert(!"non contiguous region exception wasn't caught");
  } catch (const sycl::invalid_object_error &e) {
    std::cerr << e.what() << std::endl;
  }

  try {
    sycl::buffer<int, 3> sub_buf{buf3d,
                                 /*offset*/ sycl::range<3>{0, 2, 1},
                                 /*size*/ sycl::range<3>{1, 2, 3}};
    assert(!"non contiguous region exception wasn't caught");
  } catch (const sycl::invalid_object_error &e) {
    std::cerr << e.what() << std::endl;
  }

  try {
    sycl::buffer<int, 3> sub_buf{buf3d,
                                 /*offset*/ sycl::range<3>{0, 0, 0},
                                 /*size*/ sycl::range<3>{2, 3, 4}};
    assert(!"non contiguous region exception wasn't caught");
  } catch (const sycl::invalid_object_error &e) {
    std::cerr << e.what() << std::endl;
  }

  // out of bounds
  try {
    sycl::buffer<int, 2> sub_buf{buf2d, /*offset*/ sycl::range<2>{2, 2},
                                 /*size*/ sycl::range<2>{2, 8}};
    assert(!"out of bounds exception wasn't caught");
  } catch (const sycl::invalid_object_error &e) {
    std::cerr << e.what() << std::endl;
  }

  try {
    sycl::buffer<int, 3> sub_buf{buf3d,
                                 /*offset*/ sycl::range<3>{1, 1, 1},
                                 /*size*/ sycl::range<3>{1, 1, 4}};
    assert(!"out of bounds exception wasn't caught");
  } catch (const sycl::invalid_object_error &e) {
    std::cerr << e.what() << std::endl;
  }

  try {
    sycl::buffer<int, 3> sub_buf{buf3d,
                                 /*offset*/ sycl::range<3>{3, 3, 0},
                                 /*size*/ sycl::range<3>{1, 2, 4}};
    assert(!"out of bounds exception wasn't caught");
  } catch (const sycl::invalid_object_error &e) {
    std::cerr << e.what() << std::endl;
  }

  // subbuffer from subbuffer
  try {
    sycl::buffer<int, 2> sub_buf{buf2d, /*offset*/ sycl::range<2>{2, 0},
                                 /*size*/ sycl::range<2>{2, 8}};
    sycl::buffer<int, 2> sub_sub_buf(sub_buf, sycl::range<2>{0, 0},
                                     /*size*/ sycl::range<2>{0, 0});
    assert(!"invalid subbuffer exception wasn't caught");
  } catch (const sycl::invalid_object_error &e) {
    std::cerr << e.what() << std::endl;
  }
}

void copyBlock() {
  using typename sycl::access::mode;
  using buffer = sycl::buffer<int, 1>;

  auto CopyF = [](buffer &Buffer, buffer &Block, size_t Idx, size_t BlockSize) {
    auto Subbuf = buffer(Buffer, Idx * BlockSize, BlockSize);
    auto *Src = Subbuf.get_access<mode::read>().get_pointer();
    auto *Dst = Block.get_access<mode::write>().get_pointer();
    std::copy(Src, Src + BlockSize, Dst);
  };

  try {
    static const size_t N = 100;
    static const size_t NBlock = 4;
    static const size_t BlockSize = N / NBlock;

    buffer Buffer(N);

    // Init with data
    {
      auto *Acc = Buffer.get_access<mode::write>().get_pointer();

      for (size_t Idx = 0; Idx < N; Idx++) {
        Acc[Idx] = Idx;
      }
    }

    std::vector<buffer> BlockBuffers;
    BlockBuffers.reserve(NBlock);

    // Copy block by block
    for (size_t Idx = 0; Idx < NBlock; Idx++) {
      auto InsertedIt = BlockBuffers.emplace(BlockBuffers.end(), BlockSize);
      CopyF(Buffer, *InsertedIt, Idx, BlockSize);
    }

    // Validate copies
    for (size_t Idx = 0; Idx < BlockBuffers.size(); ++Idx) {
      buffer &BlockB = BlockBuffers[Idx];

      auto *V = BlockB.get_access<mode::read>().get_pointer();

      for (size_t Idx2 = 0; Idx2 < BlockSize; ++Idx2) {
        assert(V[Idx2] == Idx2 + BlockSize * Idx &&
               "Invalid data in block buffer");
      }
    }
  } catch (sycl::exception &ex) {
    assert(false && "Unexpected exception captured!");
  }
}

void checkMultipleContexts() {
  constexpr int N = 64;
  int a[N] = {0};
  {
    sycl::queue queue1;
    sycl::queue queue2;
    sycl::buffer<int, 1> buf(a, sycl::range<1>(N));
    sycl::buffer<int, 1> subbuf1(buf, sycl::id<1>(0), sycl::range<1>(N / 2));
    sycl::buffer<int, 1> subbuf2(buf, sycl::id<1>(N / 2),
                                 sycl::range<1>(N / 2));
    queue1.submit([&](sycl::handler &cgh) {
      auto bufacc = subbuf1.get_access<sycl::access::mode::read_write>(cgh);
      cgh.parallel_for<class sub_buffer_1>(
          sycl::range<1>(N / 2), [=](sycl::id<1> idx) { bufacc[idx[0]] = 1; });
    });

    queue2.submit([&](sycl::handler &cgh) {
      auto bufacc = subbuf2.get_access<sycl::access::mode::read_write>(cgh);
      cgh.parallel_for<class sub_buffer_2>(
          sycl::range<1>(N / 2), [=](sycl::id<1> idx) { bufacc[idx[0]] = 2; });
    });
  }
  assert(a[N / 2 - 1] == 1 && a[N / 2] == 2 && "Sub buffer data loss");

  {
    sycl::queue queue1;
    sycl::buffer<int, 1> buf(a, sycl::range<1>(N));
    sycl::buffer<int, 1> subbuf1(buf, sycl::id<1>(N / 2),
                                 sycl::range<1>(N / 2));
    queue1.submit([&](sycl::handler &cgh) {
      auto bufacc = subbuf1.get_access<sycl::access::mode::read_write>(cgh);
      cgh.parallel_for<class sub_buffer_3>(
          sycl::range<1>(N / 2), [=](sycl::id<1> idx) { bufacc[idx[0]] = -1; });
    });
    auto host_acc = subbuf1.get_access<sycl::access::mode::read>();
    assert(host_acc[0] == -1 && host_acc[N / 2 - 1] == -1 &&
           "Sub buffer data loss");
  }
}

int main() {
  sycl::queue q;
  check1DSubBuffer(q);
  checkHostAccessor(q);
  checkExceptions();
  copyBlock();
  checkMultipleContexts();
  return 0;
}
