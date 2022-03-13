//===-- ctor_converting.hpp - Functions for tests on simd converting constructor
//      definition. -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides functions for tests on simd converting constructor.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "../value_conv.hpp"
#include "common.hpp"

namespace esimd = sycl::ext::intel::esimd;

namespace esimd_test::api::functional::ctors {

// Descriptor class for the case of calling constructor in initializer context.
struct initializer {
  static std::string get_description() { return "initializer"; }

  template <typename SrcT, typename DstT, int NumElems>
  static void call_simd_ctor(const SrcT *const ref_data, DstT *const out) {
    esimd::simd<SrcT, NumElems> input_simd;
    input_simd.copy_from(ref_data);

    esimd::simd<DstT, NumElems> output_simd = input_simd;
    output_simd.copy_to(out);
  }
};

// Descriptor class for the case of calling constructor in variable declaration
// context.
struct var_decl {
  static std::string get_description() { return "variable declaration"; }

  template <typename SrcT, typename DstT, int NumElems>
  static void call_simd_ctor(const SrcT *const ref_data, DstT *const out) {
    esimd::simd<SrcT, NumElems> input_simd;
    input_simd.copy_from(ref_data);

    esimd::simd<DstT, NumElems> output_simd(input_simd);
    output_simd.copy_to(out);
  }
};

// Descriptor class for the case of calling constructor in rvalue in an
// expression context.
struct rval_in_expr {
  static std::string get_description() { return "rvalue in an expression"; }

  template <typename SrcT, typename DstT, int NumElems>
  static void call_simd_ctor(const SrcT *const ref_data, DstT *const out) {
    esimd::simd<SrcT, NumElems> input_simd;
    input_simd.copy_from(ref_data);

    esimd::simd<DstT, NumElems> output_simd;
    output_simd = esimd::simd<DstT, NumElems>(input_simd);
    output_simd.copy_to(out);
  }
};

// Descriptor class for the case of calling constructor in const reference
// context.
class const_ref {
public:
  static std::string get_description() { return "const reference"; }

  template <typename SrcT, typename DstT, int NumElems>
  static void call_simd_ctor(const SrcT *const ref_data, DstT *const out) {
    esimd::simd<SrcT, NumElems> input_simd;
    input_simd.copy_from(ref_data);
    call_simd_by_const_ref<SrcT, DstT, NumElems>(
        esimd::simd<SrcT, NumElems>(input_simd), out);
  }

private:
  template <typename SrcT, typename DstT, int NumElems>
  static void
  call_simd_by_const_ref(const esimd::simd<SrcT, NumElems> &simd_by_const_ref,
                         DstT *out) {
    esimd::simd<DstT, NumElems> output_simd = simd_by_const_ref;
    output_simd.copy_to(out);
  }
};

template <typename T, int NumElems, typename ContextT>
class ConvCtorTestDescription : public ITestDescription {
public:
  ConvCtorTestDescription(size_t index, T retrieved_val, T expected_val,
                          const std::string &src_data_type,
                          const std::string &dst_data_type)
      : m_src_data_type(src_data_type), m_dst_data_type(dst_data_type),
        m_retrieved_val(retrieved_val), m_expected_val(expected_val),
        m_index(index) {}

  std::string to_string() const override {
    // TODO: Make strings for fp values more short during failure output, may be
    // by using hex representation
    std::string log_msg("Failed for converting from simd<");

    log_msg += m_src_data_type + ", " + std::to_string(NumElems) + ">";
    log_msg +=
        ", to simd<" + m_dst_data_type + ", " + std::to_string(NumElems) + ">";
    log_msg += ", with context: " + ContextT::get_description();
    log_msg += ", retrieved: " + std::to_string(m_retrieved_val);
    log_msg += ", expected: " + std::to_string(m_expected_val);
    log_msg += ", at index: " + std::to_string(m_index);

    return log_msg;
  }

private:
  const std::string m_src_data_type;
  const std::string m_dst_data_type;
  const T m_retrieved_val;
  const T m_expected_val;
  const size_t m_index;
};

// The main test routine.
// Using functor class to be able to iterate over the pre-defined data types.
template <typename SrcT, typename DimT, typename DstT, typename TestCaseT>
class run_test {
  static constexpr int NumElems = DimT::value;

public:
  bool operator()(sycl::queue &queue, const std::string &src_data_type,
                  const std::string &dst_data_type) {
    bool passed = true;
    const std::vector<SrcT> ref_data =
        generate_ref_conv_data<SrcT, DstT, NumElems>();

    // If current number of elements is equal to one, then run test with each
    // one value from reference data.
    // If current number of elements is greater than one, then run tests with
    // whole reference data.
    if constexpr (NumElems == 1) {
      for (size_t i = 0; i < ref_data.size(); ++i) {
        passed &= run_verification(queue, {ref_data[i]}, src_data_type,
                                   dst_data_type);
      }
    } else {
      passed &= run_verification(queue, ref_data, src_data_type, dst_data_type);
    }
    return passed;
  }

private:
  bool run_verification(sycl::queue &queue, const std::vector<SrcT> &ref_data,
                        const std::string &src_data_type,
                        const std::string &dst_data_type) {
    assert(ref_data.size() == NumElems &&
           "Reference data size is not equal to the simd vector length.");

    bool passed = true;

    shared_vector<DstT> result(NumElems, shared_allocator<DstT>(queue));
    shared_vector<SrcT> shared_ref_data(ref_data.begin(), ref_data.end(),
                                        shared_allocator<SrcT>(queue));

    queue.submit([&](sycl::handler &cgh) {
      const SrcT *const ref = shared_ref_data.data();
      DstT *const out = result.data();

      cgh.single_task<Kernel<SrcT, NumElems, DstT, TestCaseT>>(
          [=]() SYCL_ESIMD_KERNEL {
            TestCaseT::template call_simd_ctor<SrcT, DstT, NumElems>(ref, out);
          });
    });
    queue.wait_and_throw();

    for (size_t i = 0; i < result.size(); ++i) {
      // We ensure there is no UB here by preparing appropriate reference
      // values.
      const DstT &expected = static_cast<DstT>(ref_data[i]);
      const DstT &retrieved = result[i];
      if constexpr (type_traits::is_sycl_floating_point_v<DstT>) {
        // std::isnan() couldn't be called for integral types because it call is
        // ambiguous GitHub issue for that case:
        // https://github.com/microsoft/STL/issues/519
        if (!std::isnan(expected) || !std::isnan(retrieved)) {
          if (expected != retrieved) {
            // TODO add a function that will compare with defined accuracy
            // taking into account the possibility of UB on border values.
            // We don't have a such UB now because we are using 10f as maximum
            // value.
            if (!((expected + value<DstT>::pos_ulp(expected)) >= retrieved ||
                  (expected - value<DstT>::pos_ulp(expected)) >= retrieved ||
                  (expected + value<DstT>::pos_ulp(expected)) <= retrieved ||
                  (expected - value<DstT>::pos_ulp(expected)) <= retrieved)) {
              passed = fail_test(i, retrieved, expected, src_data_type,
                                 dst_data_type);
            }
          }
        }
      } else {
        if (expected != retrieved) {
          passed =
              fail_test(i, retrieved, expected, src_data_type, dst_data_type);
        }
      }
    }

    return passed;
  }

  bool fail_test(size_t index, DstT retrieved, DstT expected,
                 const std::string &src_data_type,
                 const std::string &dst_data_type) {
    const auto description = ConvCtorTestDescription<DstT, NumElems, TestCaseT>(
        index, retrieved, expected, src_data_type, dst_data_type);
    log::fail(description);

    return false;
  }
};

} // namespace esimd_test::api::functional::ctors
