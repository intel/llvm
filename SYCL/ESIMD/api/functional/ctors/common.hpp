//===-- common.hpp - Define common code for simd ctors tests --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides common things for simd ctors tests.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "../common.hpp"

namespace esimd_test {
namespace api {
namespace functional {
namespace ctors {

// Dummy kernel for submitting some code into device side.
template <typename DataT, int NumElems, typename T> struct Kernel;

template <typename DataT>
using shared_allocator = sycl::usm_allocator<DataT, sycl::usm::alloc::shared>;

template <typename DataT>
using shared_vector = std::vector<DataT, shared_allocator<DataT>>;

// Calls simd constructor in provided invocation context, which depends on the
// TestCaseT type. TestCaseT is a struct, that should have call_simd_ctor method
// that should return constructed object of simd class.
// This function returns std::vector instance with the output data.
template <typename DataT, int NumElems, typename TestCaseT>
auto call_simd(sycl::queue &queue, const shared_vector<DataT> &ref_data) {

  shared_vector<DataT> result{NumElems, shared_allocator<DataT>{queue}};

  queue.submit([&](sycl::handler &cgh) {
    const auto ref = ref_data.data();
    auto out = result.data();

    cgh.single_task<Kernel<DataT, NumElems, TestCaseT>>(
        [=]() SYCL_ESIMD_KERNEL {
          sycl::ext::intel::experimental::esimd::simd<DataT, NumElems>
              result_simd =
                  TestCaseT::template call_simd_ctor<DataT, NumElems>(ref);
          result_simd.copy_to(out);
        });
  });
  return result;
}

// The main test routine.
// Using functor class to be able to iterate over the pre-defined data types.
template <typename DataT, int NumElems, typename TestCaseT> struct test {
  bool operator()(sycl::queue &queue, const std::string &data_type) {
    bool passed{true};

    std::vector<DataT> generated_data{generate_ref_data<DataT, NumElems>()};
    shared_vector<DataT> ref_data{generated_data.begin(), generated_data.end(),
                                  shared_allocator<DataT>{queue}};

    const auto result_data =
        call_simd<DataT, NumElems, TestCaseT>(queue, ref_data);

    for (size_t it = 0; it < ref_data.size(); it++) {
      if (!are_bitwise_equal(ref_data[it], result_data[it])) {
        passed = false;
        log::fail<NumElems>(
            "Simd by " + TestCaseT::get_description() +
                " failed, retrieved: " + std::to_string(result_data[it]) +
                ", expected: " + std::to_string(ref_data[it]),
            data_type);
      }
    }

    return passed;
  }
};

} // namespace ctors
} // namespace functional
} // namespace api
} // namespace esimd_test
