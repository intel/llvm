//==------- ctor_load_acc_core.cpp  - DPC++ ESIMD on-device test -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu, level_zero
// XREQUIRES: gpu
// TODO gpu and level_zero in REQUIRES due to only this platforms supported yet.
// The current "REQUIRES" should be replaced with "gpu" only as mentioned in
// "XREQUIRES".
// UNSUPPORTED: cuda, hip
// RUN: %clangxx -fsycl %s -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// Test for simd load from accessor constructor.
// The test uses reference data and different alignment flags. Invokes simd
// constructors in different contexts with provided reference data and alignment
// flag using accessor as input.
// It is expected for destination simd instance to store a bitwise same data as
// the reference one.

#include "ctor_load_acc.hpp"
#include "ctor_load_acc_coverage.hpp"

using namespace esimd_test::api::functional;

int main(int, char **) {
  sycl::queue queue(esimd_test::ESIMDSelector,
                    esimd_test::createExceptionHandler());

  bool passed = true;

  // Define input for simd<T,N> combinations
  const auto types = get_tested_types<tested_types::core>();
  const auto sizes = get_all_sizes();

  // Define input for sycl::accessor<T,Dims,Mode,Target> type combinations
  const auto acc_dims = ctors::coverage::get_all_dimensions();
  const auto acc_modes = ctors::coverage::get_all_modes();
  const auto acc_targets = ctors::coverage::get_all_targets();

  // Define input for constructor calls
  // Offset generator is used to have offset to be multiplier of alignment flag
  const auto contexts = ctors::coverage::get_all_contexts();
  const auto offset_generators = ctors::coverage::get_all_offset_generators();

  // Run over all types with element and vector alignment
  // Using overaligned tag with alignment requirements appropriate for any
  // type from the type coverage
  {
    constexpr auto oword = 16;
    const auto alignments =
        named_type_pack<ctors::alignment::element, ctors::alignment::vector,
                        ctors::alignment::overal<oword>>::generate();

    // Run for all combinations possible
    passed &= for_all_combinations<ctors::run_test>(
        types, sizes, acc_dims, acc_modes, acc_targets, contexts,
        offset_generators, alignments, queue);
  }
  {
    // Smoke test for simple data load and for power-of-two requirement
    const auto single_alignment =
        named_type_pack<ctors::alignment::overal<1>>::generate();
    const auto single_type =
        named_type_pack<signed char>::generate("signed char");

    passed &= for_all_combinations<ctors::run_test>(
        single_type, sizes, acc_dims, acc_modes, acc_targets, contexts,
        offset_generators, single_alignment, queue);
  }

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
