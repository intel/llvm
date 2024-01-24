//==------------ lsc_predicate_stateless.cpp - DPC++ ESIMD on-device test -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-pvc || gpu-intel-dg2
// RUN: %{build} -fsycl-esimd-force-stateless-mem -o %t.out
// RUN: %{run} %t.out

// The test checks functionality of the lsc_block_load, lsc_block_store
// accessor - based ESIMD intrinsics when stateless memory accesses are
// enforced, i.e. accessor based accesses are automatically converted to
// stateless accesses with newly introduced predicate parameter.

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
  auto vec_2 = std::vector<int>(size);

  std::iota(vec_0.begin(), vec_0.end(), 0);
  std::iota(vec_2.begin(), vec_2.end(), 0);
  auto buf_0 = buffer{vec_0};
  auto buf_2 = buffer{vec_2};

  try {
    q.submit([&](handler &h) {
      auto access_0 = buf_0.template get_access<access::mode::read_write>(h);
      auto access_2 = buf_2.template get_access<access::mode::read_write>(h);

      h.parallel_for(
          range<1>{size / SIMDSize}, [=](id<1> id) SYCL_ESIMD_KERNEL {
            auto offset = id[0] * SIMDSize * sizeof(int);
            auto offsets = simd<uint32_t, SIMDSize>(id * SIMDSize * sizeof(int),
                                                    sizeof(int));
            auto pred_enable = simd_mask<1>(1);
            auto pred_disable = simd_mask<1>(0);

            auto data_0 =
                lsc_block_load<int, SIMDSize>(access_0, offset, pred_enable);
            lsc_block_store<int, SIMDSize>(access_0, offset, data_0 * 2,
                                           pred_enable);

            auto data_2 =
                lsc_block_load<int, SIMDSize>(access_2, offset, pred_enable);
            lsc_block_store<int, SIMDSize>(access_2, offset, data_2 * 2,
                                           pred_disable);
          });
    });
    q.wait();
    buf_0.template get_access<access::mode::read_write>();
    buf_2.template get_access<access::mode::read_write>();
  } catch (sycl::exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    return 1;
  }

  auto error = 0;
  for (auto i = 0; i != size; ++i) {
    if (vec_0[i] != 2 * i) {
      ++error;
      std::cout << " Accessor Test 1 out[" << i << "] = 0x" << std::hex
                << vec_0[i] << " vs etalon = 0x" << 2 * i << std::dec
                << std::endl;
    }

    if (vec_2[i] != i) {
      ++error;
      std::cout << " Accessor Test 2 out[" << i << "] = 0x" << std::hex
                << vec_2[i] << " vs etalon = 0x" << i << std::dec << std::endl;
    }
  }

  std::cout << "Accessor lsc predicate test ";
  std::cout << (error != 0 ? "FAILED" : "passed") << std::endl;
  return error;
}

int main() {

  auto q =
      queue{esimd_test::ESIMDSelector, esimd_test::createExceptionHandler()};
  auto device = q.get_device();
  std::cout << "Device name: " << device.get_info<info::device::name>()
            << std::endl;

  int error = testAccessor<8>(q);
  error += testAccessor<16>(q);
  error += testAccessor<32>(q);
  return error;
}
