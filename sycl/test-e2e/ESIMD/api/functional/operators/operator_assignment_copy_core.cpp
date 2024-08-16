//==------- operator_assignment_move_and_copy_core.cpp  - DPC++ ESIMD on-device
//          test -----------------------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out
//
// Test for simd copy assignment operators.
// The test creates source simd instance with reference data and invokes
// copy assignment operator from source to destination simd instance. It is
// expected for destination simd instance to store a bitwise same data as the
// reference one.

#include "operator_assignment.hpp"

using namespace sycl::ext::intel::esimd;
using namespace esimd_test::api::functional;

// Descriptor class for the case of calling copy assignment operator.
struct copy_assignment {
  static std::string get_description() { return "copy assignment operator"; }

  static constexpr bool is_move_expected() { return false; }

  template <typename DataT, int NumElems>
  static bool run(const DataT *const ref_data, DataT *const out) {
    simd<DataT, NumElems> source_simd;
    source_simd.copy_from(ref_data);
    simd<DataT, NumElems> simd_obj;
    simd_obj = source_simd;
    simd_obj.copy_to(out);
    return false;
  }
};

int main(int, char **) {
  sycl::queue queue(esimd_test::ESIMDSelector,
                    esimd_test::createExceptionHandler());

  bool passed = true;

  const auto types = get_tested_types<tested_types::core>();
  const auto all_sizes = get_all_sizes();

  const auto context = unnamed_type_pack<copy_assignment>::generate();

  passed &= for_all_combinations<operators::run_test>(types, all_sizes, context,
                                                      queue);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
