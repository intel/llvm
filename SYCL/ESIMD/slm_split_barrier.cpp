//==----------- slm_split_barrier.cpp  - DPC++ ESIMD on-device test --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// UNSUPPORTED: cuda || hip

#include "esimd_test_utils.hpp"

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace cl::sycl;
using namespace sycl::ext::intel;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

#define LOCAL_SIZE 4
#define GLOBAL_SIZE 6
#define NUM_THREADS LOCAL_SIZE *GLOBAL_SIZE

/// \brief transfer data from memory to SLM.
///
/// Load ::size bytes from memory pointer ::addr starting at ::offset to the
/// SLM ::slmOffset. ::size must be a multiple of 256.
///
ESIMD_INLINE
void load_to_slm(uint grpSize, uint localId, uint slmOffset, char *addr,
                 uint offset, uint size) {
  simd<uint, 16> vOffset(0, 16);

  uint numTotalBlocks = size / 256;
  uint numBlocks = numTotalBlocks / grpSize;
  uint numLeftOver = numTotalBlocks % grpSize;
  numBlocks += (localId < numLeftOver) ? 1 : 0;

  uint threadOffsetInSLM = slmOffset + localId * 256;
  // in bytes
  uint threadOffsetInMemory = offset + threadOffsetInSLM;
  // in unit of bytes
  simd<uint, 16> vOffsets = vOffset + threadOffsetInSLM;

  for (uint block = 0; block < numBlocks; block++) {
    simd<uint, 32> row0; // 32 floats or 128 Bytes or 4 GRF-registers
    simd<uint, 32> row1;
    simd<uint, 64> rowTrans;
    row0.copy_from((const uint *)(addr + threadOffsetInMemory));
    row1.copy_from((const uint *)(addr + threadOffsetInMemory + 128));

    // Transpose
    rowTrans.select<8, 1>(0) = row0.select<8, 4>(0);
    rowTrans.select<8, 1>(16) = row0.select<8, 4>(1);
    rowTrans.select<8, 1>(32) = row0.select<8, 4>(2);
    rowTrans.select<8, 1>(48) = row0.select<8, 4>(3);

    rowTrans.select<8, 1>(8) = row1.select<8, 4>(0);
    rowTrans.select<8, 1>(24) = row1.select<8, 4>(1);
    rowTrans.select<8, 1>(40) = row1.select<8, 4>(2);
    rowTrans.select<8, 1>(56) = row1.select<8, 4>(3);

    slm_scatter_rgba<uint, 16, rgba_channel_mask::ABGR>(vOffsets, rowTrans);
    threadOffsetInMemory += grpSize * 256;
    vOffsets += (grpSize * 256);
  }

  // add memory fence and split barriers
  fence<fence_mask::global_coherent_fence>();
  split_barrier<split_barrier_action::signal>();
  split_barrier<split_barrier_action::wait>();
}

int main(void) {
  constexpr unsigned VL = 16;
  constexpr unsigned Size = NUM_THREADS * VL;

  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
  uint *A = malloc_shared<uint>(Size, q);
  uint *B = malloc_shared<uint>(Size, q);

  // Checking with specific inputs
  for (int i = 0; i < NUM_THREADS; i++) {
    uint *A_int = (uint *)(A + i * VL);
    for (int j = 0; j < VL; j++) {
      A_int[j] = i + j;
      std::cout << A_int[j] << " ";
    }
    std::cout << std::endl;
  }

  // We need that many workitems
  cl::sycl::range<1> GlobalRange{GLOBAL_SIZE};

  // Number of workitems in a workgroup
  cl::sycl::range<1> LocalRange{LOCAL_SIZE};
  cl::sycl::nd_range<1> Range{GlobalRange * LocalRange, LocalRange};

  try {
    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for<class Test>(
          Range, [=](cl::sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {
            simd<uint, VL> v_slmData;
            simd<uint, VL> v_Off(0, 4);

            uint localID = ndi.get_local_id(0);
            uint groupSize = ndi.get_local_range(0);
            uint globalID = ndi.get_global_id(0);
            uint groupID = ndi.get_group(0);

            slm_init<1024>();

            int grpMemOffset = groupID * groupSize * VL * 4;

            load_to_slm(groupSize, localID, 0, (char *)A, grpMemOffset,
                        groupSize * VL * 4);

            auto shiftID = (localID + 1) % 4;

            v_Off = v_Off + shiftID * 64;

            v_slmData = slm_gather<uint, VL>(v_Off);

            v_slmData.copy_to(B + globalID * VL);
          });
    });
    e.wait();
  } catch (cl::sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    sycl::free(A, q);
    sycl::free(B, q);
    return e.code().value();
  }

  std::cout << "result" << std::endl;
  int result = 0;
  for (int i = 0; i < NUM_THREADS; i++) {
    unsigned int *p = (unsigned int *)(B + i * VL);
    if ((i % 4) != 3) {
      for (int j = 0; j < VL; j++) {
        std::cout << (*p) << " ";
        if (*p != (i + 1 + j)) {
          result = -1;
        }
        p++;
      }
    } else {
      for (int j = 0; j < VL; j++) {
        std::cout << (*p) << " ";
        if (*p != (i - 3 + j)) {
          result = -1;
        }
        p++;
      }
    }
    std::cout << std::endl;
  }
  sycl::free(A, q);
  sycl::free(B, q);

  std::cout << (result < 0 ? "FAILED\n" : "Passed\n");
  return result < 0 ? 1 : 0;
}
