//==------- ctor_default.cpp  - DPC++ ESIMD on-device test ----------------==//
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
// Test for esimd default constructor.
//
// The simd can't be constructed with sycl::half data type. The issue was
// created (https://github.com/intel/llvm/issues/5077) and the TEST_HALF macros
// must be enabled when it is resolved.

#include "../common.hpp"

using namespace cl::sycl;
using namespace sycl::ext::intel::experimental::esimd;
using namespace esimd_test::api::functional;

// Dummy kernel for submitting some code into device side.
template <typename DataT, int NumElems, typename T> struct Kernel;

// Descriptor class for the case of calling constructor in initializer context
struct initializer {
  static std::string get_description() { return "initializer"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(DataT *output_data) {
    simd<DataT, NumElems> simd_by_init = simd<DataT, NumElems>();
    simd_by_init.copy_to(output_data);
  }
};

// Descriptor class for the case of calling constructor in variable declaration
// context
struct var_declaration {
  static std::string get_description() { return "variable declaration"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(DataT *output_data) {
    simd<DataT, NumElems> simd_by_var_decl;
    simd_by_var_decl.copy_to(output_data);
  }
};

// Descriptor class for the case of calling constructor in rvalue in an
// expression context
struct rval_in_expression {
  static std::string get_description() { return "rvalue in an expression"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(DataT *output_data) {
    simd<DataT, NumElems> simd_by_rval;
    simd_by_rval = simd<DataT, NumElems>();
    simd_by_rval.copy_to(output_data);
  }
};

// Descriptor class for the case of calling constructor in const reference
// context
struct const_ref {
  static std::string get_description() { return "const reference"; }

  template <typename DataT, int NumElems>
  static void
  call_simd_by_const_ref(const simd<DataT, NumElems> &simd_by_const_ref,
                         DataT *output_data) {
    simd_by_const_ref.copy_to(output_data);
  }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(DataT *output_data) {
    call_simd_by_const_ref<DataT, NumElems>(simd<DataT, NumElems>(),
                                            output_data);
  }
};

// Functor with the main test routine to iterate over the pre-determined
// datatypes
template <typename DataT, int NumElems, typename TestCaseT> struct test {
  bool operator()(sycl::queue &queue, const std::string &data_type) {
    bool passed{true};

    DataT default_val{};
    using AllocatorT = sycl::usm_allocator<DataT, sycl::usm::alloc::shared>;

    std::vector<DataT, AllocatorT> result{NumElems, AllocatorT{queue}};
    queue.submit([&](sycl::handler &cgh) {
      auto out = result.data();
      cgh.single_task<Kernel<DataT, NumElems, TestCaseT>>(
          [=]() SYCL_ESIMD_KERNEL {
            TestCaseT::template call_simd_ctor<DataT, NumElems>(out);
          });
    });
    for (const auto &it : result) {
      if (it != default_val) {
        passed = false;
        log::fail<NumElems>(
            "In simd by " + TestCaseT::get_description() +
                " elem value is not equal to default value, retrieved: " +
                std::to_string(it) +
                ", expected: " + std::to_string(default_val),
            data_type);
      }
    }
    return passed;
  }
};

template <typename DataT, typename TestT>
using run_test_with_one_elem = test<DataT, 1, TestT>;

template <typename DataT, typename TestT>
using run_test_with_eight_elems = test<DataT, 8, TestT>;

template <typename DataT, typename TestT>
using run_test_with_sixteen_elems = test<DataT, 16, TestT>;

template <typename DataT, typename TestT>
using run_test_with_thirty_two_elems = test<DataT, 32, TestT>;

template <typename TestT, typename... T>
bool run_verification_with_chosen_test_type(
    sycl::queue &queue, const named_type_pack<T...> &types) {
  bool passed{true};

  passed &= for_all_types<run_test_with_one_elem, TestT>(types, queue);
  passed &= for_all_types<run_test_with_eight_elems, TestT>(types, queue);
  passed &= for_all_types<run_test_with_sixteen_elems, TestT>(types, queue);
  passed &= for_all_types<run_test_with_thirty_two_elems, TestT>(types, queue);
  return passed;
}

int main(int argc, char **argv) {
  sycl::queue queue{esimd_test::ESIMDSelector{},
                    esimd_test::createExceptionHandler()};

  bool passed{true};

  const auto types{get_tested_types<tested_types::all>()};

  passed &= run_verification_with_chosen_test_type<initializer>(queue, types);
  passed &=
      run_verification_with_chosen_test_type<var_declaration>(queue, types);
  passed &=
      run_verification_with_chosen_test_type<rval_in_expression>(queue, types);
  passed &= run_verification_with_chosen_test_type<const_ref>(queue, types);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
