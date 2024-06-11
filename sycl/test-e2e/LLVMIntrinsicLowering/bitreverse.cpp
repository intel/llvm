// Test that llvm.bitreverse is lowered correctly by llvm-spirv.

// UNSUPPORTED: hip || cuda

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

// CHECK-SPV: Name {{[0-9]+}} "llvm_bitreverse_i8"
// CHECK-SPV: Name {{[0-9]+}} "llvm_bitreverse_i16"
// CHECK-SPV: Name {{[0-9]+}} "llvm_bitreverse_i32"
// CHECK-SPV: Name {{[0-9]+}} "llvm_bitreverse_i64"

// CHECK-SPV: Name {{[0-9]+}} "llvm_bitreverse_v2i8"
// CHECK-SPV: Name {{[0-9]+}} "llvm_bitreverse_v2i16"
// CHECK-SPV: Name {{[0-9]+}} "llvm_bitreverse_v2i32"

// CHECK-SPV: Name {{[0-9]+}} "llvm_bitreverse_v3i8"
// CHECK-SPV: Name {{[0-9]+}} "llvm_bitreverse_v3i16"
// CHECK-SPV: Name {{[0-9]+}} "llvm_bitreverse_v3i32"

// CHECK-SPV: Name {{[0-9]+}} "llvm_bitreverse_v4i8"
// CHECK-SPV: Name {{[0-9]+}} "llvm_bitreverse_v4i16"
// CHECK-SPV: Name {{[0-9]+}} "llvm_bitreverse_v4i32"

// CHECK-SPV: Name {{[0-9]+}} "llvm_bitreverse_v8i8"
// CHECK-SPV: Name {{[0-9]+}} "llvm_bitreverse_v8i16"
// CHECK-SPV: Name {{[0-9]+}} "llvm_bitreverse_v8i32"

// CHECK-SPV: Name {{[0-9]+}} "llvm_bitreverse_v16i8"
// CHECK-SPV: Name {{[0-9]+}} "llvm_bitreverse_v16i16"
// CHECK-SPV: Name {{[0-9]+}} "llvm_bitreverse_v16i32"

// CHECK-SPV: LinkageAttributes "llvm_bitreverse_i8" Export
// CHECK-SPV: LinkageAttributes "llvm_bitreverse_i16" Export
// CHECK-SPV: LinkageAttributes "llvm_bitreverse_i32" Export
// CHECK-SPV: LinkageAttributes "llvm_bitreverse_i64" Export

// CHECK-SPV: LinkageAttributes "llvm_bitreverse_v2i8" Export
// CHECK-SPV: LinkageAttributes "llvm_bitreverse_v2i16" Export
// CHECK-SPV: LinkageAttributes "llvm_bitreverse_v2i32" Export

// CHECK-SPV: LinkageAttributes "llvm_bitreverse_v3i8" Export
// CHECK-SPV: LinkageAttributes "llvm_bitreverse_v3i16" Export
// CHECK-SPV: LinkageAttributes "llvm_bitreverse_v3i32" Export

// CHECK-SPV: LinkageAttributes "llvm_bitreverse_v4i8" Export
// CHECK-SPV: LinkageAttributes "llvm_bitreverse_v4i16" Export
// CHECK-SPV: LinkageAttributes "llvm_bitreverse_v4i32" Export

// CHECK-SPV: LinkageAttributes "llvm_bitreverse_v8i8" Export
// CHECK-SPV: LinkageAttributes "llvm_bitreverse_v8i16" Export
// CHECK-SPV: LinkageAttributes "llvm_bitreverse_v8i32" Export

// CHECK-SPV: LinkageAttributes "llvm_bitreverse_v16i8" Export
// CHECK-SPV: LinkageAttributes "llvm_bitreverse_v16i16" Export
// CHECK-SPV: LinkageAttributes "llvm_bitreverse_v16i32" Export

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
  if (bitlength == 8) {
    // Avoid bug with __builtin_elementwise_bitreverse(a) on scalar 8-bit types.
    a = ((0x55 & a) << 1) | (0x55 & (a >> 1));
    a = ((0x33 & a) << 2) | (0x33 & (a >> 2));
    return (a << 4) | (a >> 4);
  } else if (bitlength == 16) {
    // Avoid bug with __builtin_elementwise_bitreverse(a) on scalar 16-bit
    // types.
    a = ((0x5555 & a) << 1) | (0x5555 & (a >> 1));
    a = ((0x3333 & a) << 2) | (0x3333 & (a >> 2));
    a = ((0x0F0F & a) << 4) | (0x0F0F & (a >> 4));
    return (a << 8) | (a >> 8);
  } else
    return __builtin_elementwise_bitreverse(a);
}

template <class T> class BitreverseTest;

#define NUM_TESTS 1024

template <typename TYPE> void do_scalar_bitreverse_test() {
  queue q;

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
    if (Output[i] != reference_reverse(Input[i], sizeof(TYPE) * 8)) {
      std::cerr << "Failed for scalar " << std::hex << Input[i]
                << " sizeof=" << sizeof(TYPE) << "\n";
      exit(-1);
    }

  free(Input, q.get_context());
  free(Output, q.get_context());
}

template <typename VTYPE> void do_vector_bitreverse_test() {
  queue q;

  VTYPE *Input = (VTYPE *)malloc_shared(sizeof(VTYPE) * NUM_TESTS,
                                        q.get_device(), q.get_context());
  VTYPE *Output = (VTYPE *)malloc_shared(sizeof(VTYPE) * NUM_TESTS,
                                         q.get_device(), q.get_context());

  for (unsigned i = 0; i < NUM_TESTS; i++)
    for (unsigned j = 0; j < __builtin_vectorelements(VTYPE); j++)
      Input[i][j] =
          get_rand<typename std::decay<decltype(Input[0][0])>::type>();

  q.submit([=](handler &cgh) {
    cgh.single_task<BitreverseTest<VTYPE>>([=]() {
      for (unsigned i = 0; i < NUM_TESTS; i++)
        Output[i] = reverse(Input[i], sizeof(Input[0][0]) * 8);
    });
  });
  q.wait();
  for (unsigned i = 0; i < NUM_TESTS; i++) {
    auto Reference = reference_reverse(Input[i], sizeof(Input[0][0]) * 8);
    for (unsigned j = 0; j < __builtin_vectorelements(VTYPE); j++)
      if (Output[i][j] != Reference[j]) {
        std::cerr << "Failed for vector " << std::hex << Input[i][j]
                  << " sizeof=" << sizeof(Input[0][0])
                  << " elements=" << __builtin_vectorelements(VTYPE) << "\n";
        exit(-1);
      }
  }
  free(Input, q.get_context());
  free(Output, q.get_context());
}

using uint8_t2 = uint8_t __attribute__((ext_vector_type(2)));
using uint16_t2 = uint16_t __attribute__((ext_vector_type(2)));
using uint32_t2 = uint32_t __attribute__((ext_vector_type(2)));
using uint64_t2 = uint64_t __attribute__((ext_vector_type(2)));

using uint8_t3 = uint8_t __attribute__((ext_vector_type(3)));
using uint16_t3 = uint16_t __attribute__((ext_vector_type(3)));
using uint32_t3 = uint32_t __attribute__((ext_vector_type(3)));
using uint64_t3 = uint64_t __attribute__((ext_vector_type(3)));

using uint8_t4 = uint8_t __attribute__((ext_vector_type(4)));
using uint16_t4 = uint16_t __attribute__((ext_vector_type(4)));
using uint32_t4 = uint32_t __attribute__((ext_vector_type(4)));
using uint64_t4 = uint64_t __attribute__((ext_vector_type(4)));

using uint8_t8 = uint8_t __attribute__((ext_vector_type(8)));
using uint16_t8 = uint16_t __attribute__((ext_vector_type(8)));
using uint32_t8 = uint32_t __attribute__((ext_vector_type(8)));
using uint64_t8 = uint64_t __attribute__((ext_vector_type(8)));

using uint8_t16 = uint8_t __attribute__((ext_vector_type(16)));
using uint16_t16 = uint16_t __attribute__((ext_vector_type(16)));
using uint32_t16 = uint32_t __attribute__((ext_vector_type(16)));
using uint64_t16 = uint64_t __attribute__((ext_vector_type(16)));

int main() {
  srand(2024);

  do_scalar_bitreverse_test<uint8_t>();
  do_scalar_bitreverse_test<uint16_t>();
  do_scalar_bitreverse_test<uint32_t>();
  do_scalar_bitreverse_test<uint64_t>();

  do_vector_bitreverse_test<uint8_t2>();
  do_vector_bitreverse_test<uint16_t2>();
  do_vector_bitreverse_test<uint32_t2>();

  do_vector_bitreverse_test<uint8_t3>();
  do_vector_bitreverse_test<uint16_t3>();
  do_vector_bitreverse_test<uint32_t3>();

  do_vector_bitreverse_test<uint8_t4>();
  do_vector_bitreverse_test<uint16_t4>();
  do_vector_bitreverse_test<uint32_t4>();

  do_vector_bitreverse_test<uint8_t8>();
  do_vector_bitreverse_test<uint16_t8>();
  do_vector_bitreverse_test<uint32_t8>();

  do_vector_bitreverse_test<uint8_t16>();
  do_vector_bitreverse_test<uint16_t16>();
  do_vector_bitreverse_test<uint32_t16>();

  return 0;
}
