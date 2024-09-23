// Test that llvm.bitreverse is lowered correctly by llvm-spirv for 2/4-bit
// types.

// UNSUPPORTED: hip || cuda

// TODO: Remove XFAIL after fixing
// https://github.com/intel/intel-graphics-compiler/issues/330
// XFAIL: gpu

// Make dump directory.
// RUN: rm -rf %t.spvdir && mkdir %t.spvdir

// Ensure that SPV_KHR_bit_instructions is disabled so that translator
// will lower llvm.bitreverse.* intrinsics instead of relying on SPIRV
// BitReverse instruction.
// Also build executable with SPV dump.
// RUN: %{build} -o %t.out -O2 -Xspirv-translator --spirv-ext=-SPV_KHR_bit_instructions -fsycl-dump-device-code=%t.spvdir

// Rename SPV file to explictly known filename.
// RUN: mv %t.spvdir/*.spv %t.spvdir/dump.spv

// Convert to text.
// RUN: llvm-spirv -to-text %t.spvdir/dump.spv

// Check that all lowerings are done by llvm-spirv.
// RUN: cat %t.spvdir/dump.spt | FileCheck %s --check-prefix CHECK-SPV --implicit-check-not=BitReverse

// Execute to ensure lowering has correct functionality.
// RUN: %{run} %t.out

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Build without lowering explicitly disabled.
// RUN: %{build} -o %t.bitinstructions.out

// Execution should still be correct.
// RUN: %{run} %t.bitinstructions.out

// CHECK-SPV: Name {{[0-9]+}} "llvm_bitreverse_i2"
// CHECK-SPV: Name {{[0-9]+}} "llvm_bitreverse_i4"

// CHECK-SPV: LinkageAttributes "llvm_bitreverse_i2" Export
// CHECK-SPV: LinkageAttributes "llvm_bitreverse_i4" Export

#include "common.hpp"
#include <iostream>
#include <string.h>
#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

using namespace sycl;

template <typename TYPE>
__attribute__((optnone, noinline)) TYPE reference_reverse(TYPE a,
                                                          const int bitlength) {
  TYPE ret = 0;
  for (auto i = 0; i < bitlength; i++) {
    ret <<= 1;
    ret |= a & 0x1;
    a >>= 1;
  }
  return ret;
}

template <typename TYPE>
__attribute__((noinline)) TYPE reverse(TYPE a, int bitlength) {
  return __builtin_elementwise_bitreverse(a);
}

template <class T> class BitreverseTest;

#define NUM_TESTS 1024

template <typename TYPE> void do_scalar_bitreverse_test() {
  queue q;

  // calculate bitlength
  int bitlength = 0;
  TYPE t = 1;
  do {
    ++bitlength;
    t <<= 1;
  } while (t);

  TYPE *Input = (TYPE *)malloc_shared(sizeof(TYPE) * NUM_TESTS, q.get_device(),
                                      q.get_context());
  TYPE *Output = (TYPE *)malloc_shared(sizeof(TYPE) * NUM_TESTS, q.get_device(),
                                       q.get_context());

  for (unsigned i = 0; i < NUM_TESTS; i++)
    Input[i] = get_rand<TYPE>();
  q.submit([=](handler &cgh) {
    cgh.single_task<BitreverseTest<TYPE>>([=]() {
      for (unsigned i = 0; i < NUM_TESTS; i++)
        Output[i] = reverse(Input[i], sizeof(TYPE) * 8);
    });
  });
  q.wait();
  for (unsigned i = 0; i < NUM_TESTS; i++)
    if (Output[i] != reference_reverse(Input[i], bitlength)) {
      std::cerr << "Failed for scalar " << std::hex
                << static_cast<uint64_t>(Input[i]) << " bitlength=" << bitlength
                << "\n";

      exit(-1);
    }

  free(Input, q.get_context());
  free(Output, q.get_context());
}

using uint2_t = _BitInt(2);
using uint4_t = _BitInt(4);

int main() {
  srand(2024);

  do_scalar_bitreverse_test<uint2_t>();
  do_scalar_bitreverse_test<uint4_t>();

  return 0;
}
