//==-------- lsc_argument_type_deduction.cpp - DPC++ ESIMD on-device test --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-pvc || esimd_emulator
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// The test checks that compiler is able to corrctly deduce the template
// parameter types for lsc functions.

#include "../esimd_test_utils.hpp"

#include <numeric>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

template <unsigned SIMDSize> int testAccessor(queue q) {
  auto size = size_t{128};

  auto vec_0 = std::vector<int>(size);

  std::iota(vec_0.begin(), vec_0.end(), 0);
  auto buf_0 = buffer{vec_0};

  try {
    q.submit([&](handler &h) {
      auto access_0 = buf_0.template get_access<access::mode::read_write>(h);

      h.parallel_for(
          range<1>{size / SIMDSize}, [=](id<1> id) SYCL_ESIMD_KERNEL {
            auto offset = id[0] * SIMDSize * sizeof(int);
            auto offsets = simd<uint32_t, SIMDSize>(id * SIMDSize * sizeof(int),
                                                    sizeof(int));
            auto data_0 = lsc_block_load<int, SIMDSize>(access_0, offset);
            lsc_block_store(access_0, offset, data_0 * 2);
          });
    });
    q.wait();
    buf_0.template get_access<access::mode::read_write>();
  } catch (sycl::exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    return 1;
  }

  auto error = 0;
  for (auto i = 0; i != size; ++i) {
    if (vec_0[i] != 2 * i) {
      ++error;
      std::cout << "out[" << i << "] = 0x" << std::hex << vec_0[i]
                << " vs etalon = 0x" << 2 * i << std::dec << std::endl;
    }
  }
  std::cout << "Accessor lsc argument type deduction test ";
  std::cout << (error != 0 ? "FAILED" : "passed") << std::endl;
  return error;
}

template <unsigned SIMDSize> int testUSM(queue q) {
  auto size = size_t{128};

  auto *vec_0 = malloc_shared<int>(size, q);
  std::iota(vec_0, vec_0 + size, 0);

  try {
    q.submit([&](handler &h) {
      h.parallel_for(
          range<1>{size / SIMDSize}, [=](id<1> id) SYCL_ESIMD_KERNEL {
            auto offset = id[0] * SIMDSize;
            auto offsets = simd<uint32_t, SIMDSize>(id * SIMDSize * sizeof(int),
                                                    sizeof(int));
            auto data_0 = lsc_block_load<int, SIMDSize>(vec_0 + offset);
            lsc_block_store(vec_0 + offset, data_0 * 2);
          });
    });
    q.wait();
  } catch (sycl::exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    sycl::free(vec_0, q);
    return 1;
  }

  int error = 0;
  for (auto i = 0; i != size; ++i) {
    if (vec_0[i] != 2 * i) {
      ++error;
      std::cout << "out[" << i << "] = 0x" << std::hex << vec_0[i]
                << " vs etalon = 0x" << 2 * i << std::dec << std::endl;
    }
  }
  sycl::free(vec_0, q);
  std::cout << "USM lsc argument type deduction test ";
  std::cout << (error != 0 ? "FAILED" : "passed") << std::endl;
  return error;
}

int main() {

  queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());
  auto device = q.get_device();
  std::cout << "Device name: " << device.get_info<info::device::name>()
            << std::endl;

  int error = testUSM<8>(q);
  error += testUSM<16>(q);
  error += testUSM<32>(q);

  error += testAccessor<8>(q);
  error += testAccessor<16>(q);
  error += testAccessor<32>(q);
  return error;
}
