//==-------- config_2d_mem_access_test.cpp  - ESIMD hardware dispatch test -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This is basic test to test hardware dispatch functionality with ESIMD.

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>
#include <vector>

using shared_allocator =
    sycl::usm_allocator<uint32_t, sycl::usm::alloc::shared>;
using shared_vector = std::vector<uint32_t, shared_allocator>;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

int main() {
  constexpr uint32_t BlockWidth = 5;
  constexpr uint32_t BlockHeight = 6;
  constexpr uint32_t NumBlocks = 7;
  constexpr uint32_t SurfaceHeight = 1;
  constexpr uint32_t SurfaceWidth = 0;
  constexpr uint32_t SurfacePitch = 2;
  constexpr uint32_t x = 3;
  constexpr uint32_t y = 4;

  sycl::queue q;
  shared_allocator allocator(q);

  shared_vector vec(9, allocator);
  {
    uint32_t *output_ptr = vec.data();
    q.single_task([=]() SYCL_ESIMD_KERNEL {
      simd<uint32_t, 9> result;
      config_2d_mem_access<uint32_t, BlockWidth, BlockHeight, NumBlocks>
          payload;
      payload.set_data_pointer(output_ptr);
      payload.set_surface_width(SurfaceWidth);
      payload.set_surface_height(SurfaceHeight);
      payload.set_surface_pitch(SurfacePitch);
      payload.set_x(x);
      payload.set_y(y);
      result[0] = payload.get_surface_width();
      result[1] = payload.get_surface_height();
      result[2] = payload.get_surface_pitch();
      result[3] = payload.get_x();
      result[4] = payload.get_y();
      result[5] = payload.get_width();
      result[6] = payload.get_height();
      result[7] = payload.get_number_of_blocks();
      auto p = payload.get_data_pointer();
      if (p == output_ptr) {
        result[8] = 8;
      } else {
        result[8] = 0;
      }

      result.copy_to(output_ptr);
    });
  }
  q.wait();

  for (int i = 0; i < vec.size(); ++i) {
    if (vec[i] != i) {
      std::cout << "Test failed for index " << i << std::endl;
      return 1;
    }
  }
  std::cout << "Pass" << std::endl;
  return 0;
}
