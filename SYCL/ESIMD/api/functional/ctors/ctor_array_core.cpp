//==------- ctor_array_core.cpp  - DPC++ ESIMD on-device test --------------==//
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
// TODO Unexpected runtime error "error: unsupported type for load/store" while
// try to use simd::copy_from(), then simd::copy_to() with fixed-size array that
// was defined on device side and the SIMD_RUN_TEST_WITH_VECTOR_LEN_1 macros
// must be enabled when it is resolved.
//
// TODO This test disabled due to simd<short, 32> vector filled with unexpected
// values from 16th element. The issue was created
// https://github.com/intel/llvm/issues/5245 and and the
// SIMD_RUN_TEST_WITH_VECTOR_LEN_32 macros must be enabled when it is resolved.
//
// Test for simd constructor from an array.
// This test uses different data types, dimensionality and different simd
// constructor invocation contexts.
// The test does the following actions:
//  - construct fixed-size array that filled with reference values
//  - use std::move() to provide it to simd constructor
//  - bitwise compare expected and retrieved values

#include "common.hpp"

using namespace esimd_test::api::functional;
using namespace sycl::ext::intel::experimental::esimd;

// Descriptor class for the case of calling constructor in initializer context.
struct initializer {
  static std::string get_description() { return "initializer"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(DataT (&&ref_data)[NumElems], DataT *const out) {
    static_assert(
        type_traits::is_nonconst_rvalue_reference_v<decltype(ref_data)>,
        "Provided input data is not nonconst rvalue reference");
    const auto simd_by_init = simd<DataT, NumElems>(std::move(ref_data));
    simd_by_init.copy_to(out);
  }
};

// Descriptor class for the case of calling constructor in variable declaration
// context.
struct var_decl {
  static std::string get_description() { return "variable declaration"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(DataT (&&ref_data)[NumElems], DataT *const out) {
    static_assert(
        type_traits::is_nonconst_rvalue_reference_v<decltype(ref_data)>,
        "Provided input data is not nonconst rvalue reference");
    const simd<DataT, NumElems> simd_by_var_decl(std::move(ref_data));
    simd_by_var_decl.copy_to(out);
  }
};

// Descriptor class for the case of calling constructor in rvalue in an
// expression context.
struct rval_in_expr {
  static std::string get_description() { return "rvalue in an expression"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(DataT (&&ref_data)[NumElems], DataT *const out) {
    static_assert(
        type_traits::is_nonconst_rvalue_reference_v<decltype(ref_data)>,
        "Provided input data is not nonconst rvalue reference");
    simd<DataT, NumElems> simd_by_rval;
    simd_by_rval = simd<DataT, NumElems>(std::move(ref_data));
    simd_by_rval.copy_to(out);
  }
};

// Descriptor class for the case of calling constructor in const reference
// context.
class const_ref {
public:
  static std::string get_description() { return "const reference"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(DataT (&&ref_data)[NumElems], DataT *const out) {
    static_assert(
        type_traits::is_nonconst_rvalue_reference_v<decltype(ref_data)>,
        "Provided input data is not nonconst rvalue reference");
    call_simd_by_const_ref<DataT, NumElems>(
        simd<DataT, NumElems>(std::move(ref_data)), out);
  }

private:
  template <typename DataT, int NumElems>
  static void
  call_simd_by_const_ref(const simd<DataT, NumElems> &simd_by_const_ref,
                         DataT *out) {
    simd_by_const_ref.copy_to(out);
  }
};

// The main test routine.
// Using functor class to be able to iterate over the pre-defined data types.
template <typename DataT, int NumElems, typename TestCaseT> class run_test {
public:
  bool operator()(sycl::queue &queue, const std::string &data_type) {

    bool passed = true;
    const std::vector<DataT> ref_data = generate_ref_data<DataT, NumElems>();

    // If current number of elements is equal to one, then run test with each
    // one value from reference data.
    // If current number of elements is greater than one, then run tests with
    // whole reference data.
    if constexpr (NumElems == 1) {
      for (size_t i = 0; i < ref_data.size(); ++i) {
        passed &= run_verification(queue, {ref_data[i]}, data_type);
      }
    } else {
      passed &= run_verification(queue, ref_data, data_type);
    }
    return passed;
  }

private:
  bool run_verification(sycl::queue &queue, const std::vector<DataT> &ref_data,
                        const std::string &data_type) {
    assert(ref_data.size() == NumElems &&
           "Reference data size is not equal to the simd vector length.");

    bool passed = true;

    shared_allocator<DataT> allocator(queue);
    shared_vector<DataT> result(NumElems, allocator);
    shared_vector<DataT> shared_ref_data(ref_data.begin(), ref_data.end(),
                                         allocator);

    queue.submit([&](sycl::handler &cgh) {
      const DataT *const ref = shared_ref_data.data();
      DataT *const out = result.data();

      cgh.single_task<ctors::Kernel<DataT, NumElems, TestCaseT>>(
          [=]() SYCL_ESIMD_KERNEL {
            DataT ref_on_dev[NumElems];
            for (size_t i = 0; i < NumElems; ++i) {
              ref_on_dev[i] = ref[i];
            }

            TestCaseT::template call_simd_ctor<DataT, NumElems>(
                std::move(ref_on_dev), out);
          });
    });
    queue.wait_and_throw();

    for (size_t i = 0; i < result.size(); ++i) {
      if (!are_bitwise_equal(ref_data[i], result[i])) {
        passed = false;

        const auto description =
            ctors::TestDescription<DataT, NumElems, TestCaseT>(
                i, result[i], ref_data[i], data_type);
        log::fail(description);
      }
    }

    return passed;
  }
};

int main(int, char **) {
  sycl::queue queue(esimd_test::ESIMDSelector{},
                    esimd_test::createExceptionHandler());

  bool passed = true;

  const auto types = get_tested_types<tested_types::all>();
#if defined(SIMD_RUN_TEST_WITH_VECTOR_LEN_1) &&                                \
    defined(SIMD_RUN_TEST_WITH_VECTOR_LEN_32)
  const auto dims = get_all_dimensions();
#elif SIMD_RUN_TEST_WITH_VECTOR_LEN_32
  const auto dims = values_pack<8, 16, 32>();
#elif SIMD_RUN_TEST_WITH_VECTOR_LEN_1
  const auto dims = values_pack<1, 8, 16>();
#else
  const auto dims = values_pack<8, 16>();
#endif

  // Run for specific combinations of types, vector length, and invocation
  // contexts.
  passed &= for_all_types_and_dims<run_test, initializer>(types, dims, queue);
  passed &= for_all_types_and_dims<run_test, var_decl>(types, dims, queue);
  passed &= for_all_types_and_dims<run_test, rval_in_expr>(types, dims, queue);
  passed &= for_all_types_and_dims<run_test, const_ref>(types, dims, queue);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
