//===-- ctor_broadcast.hpp - Functions for tests on simd broadcast constructor
//      definition. -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides functions for tests on simd broadcast constructor.
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
  static void call_simd_ctor(SrcT ref_value, DstT *const out) {
    const auto simd_by_init = esimd::simd<DstT, NumElems>(ref_value);
    simd_by_init.copy_to(out);
  }
};

// Descriptor class for the case of calling constructor in variable declaration
// context.
struct var_decl {
  static std::string get_description() { return "variable declaration"; }

  template <typename SrcT, typename DstT, int NumElems>
  static void call_simd_ctor(SrcT ref_value, DstT *const out) {
    const esimd::simd<DstT, NumElems> simd_by_var_decl(ref_value);
    simd_by_var_decl.copy_to(out);
  }
};

// Descriptor class for the case of calling constructor in rvalue in an
// expression context.
struct rval_in_expr {
  static std::string get_description() { return "rvalue in an expression"; }

  template <typename SrcT, typename DstT, int NumElems>
  static void call_simd_ctor(SrcT ref_value, DstT *const out) {
    esimd::simd<DstT, NumElems> simd_by_rval;
    simd_by_rval = esimd::simd<DstT, NumElems>(ref_value);
    simd_by_rval.copy_to(out);
  }
};

// Descriptor class for the case of calling constructor in const reference
// context.
class const_ref {
public:
  static std::string get_description() { return "const reference"; }

  template <typename SrcT, typename DstT, int NumElems>
  static void call_simd_ctor(SrcT ref_value, DstT *const out) {
    call_simd_by_const_ref<DstT, NumElems>(
        esimd::simd<DstT, NumElems>(ref_value), out);
  }

private:
  template <typename DstT, int NumElems>
  static void
  call_simd_by_const_ref(const esimd::simd<DstT, NumElems> &simd_by_const_ref,
                         DstT *const out) {
    simd_by_const_ref.copy_to(out);
  }
};

template <typename SrcT, typename DstT, int NumElems, typename ContextT>
class BroadcastCtorTestDescription : public ITestDescription {
public:
  BroadcastCtorTestDescription(size_t index, DstT retrieved_val,
                               DstT expected_val, SrcT ref_value,
                               const std::string &src_data_type,
                               const std::string &dst_data_type)
      : m_src_data_type(src_data_type), m_dst_data_type(dst_data_type),
        m_retrieved_val(retrieved_val), m_expected_val(expected_val),
        m_ref_val(ref_value), m_index(index) {}

  std::string to_string() const override {
    // TODO: Make strings for fp values more short during failure output, may be
    // by using hex representation
    std::string log_msg("Failed for simd<");

    log_msg += m_dst_data_type + ", " + std::to_string(NumElems) + ">";
    log_msg += ", with context: " + ContextT::get_description();
    log_msg += ", source type: " + m_src_data_type;
    log_msg += ", destination type: " + m_dst_data_type;
    log_msg += ", retrieved: " + std::to_string(m_retrieved_val);
    log_msg += ", expected: " + std::to_string(m_expected_val);
    log_msg += ", input: " + std::to_string(m_ref_val);
    log_msg += ", at index: " + std::to_string(m_index);

    return log_msg;
  }

private:
  const std::string m_src_data_type;
  const std::string m_dst_data_type;
  const DstT m_retrieved_val;
  const DstT m_ref_val;
  const DstT m_expected_val;
  const size_t m_index;
};

// The main test routine.
// Using functor class to be able to iterate over the pre-defined data types.
template <typename UsePositiveValueOnly, typename SrcT, typename SizeT,
          typename DstT, typename TestCaseT>
class run_test {
  static constexpr int NumElems = SizeT::value;

public:
  bool operator()(sycl::queue &queue, const std::string &src_data_type,
                  const std::string &dst_data_type) {
    bool passed = true;
    std::vector<SrcT> ref_data;

    if constexpr (UsePositiveValueOnly::value) {
      ref_data.push_back(static_cast<SrcT>(126.75));
    } else {
      ref_data = generate_ref_conv_data<SrcT, DstT, 1>();
    }

    for (size_t i = 0; i < ref_data.size(); ++i) {
      passed &=
          run_verification(queue, ref_data[i], src_data_type, dst_data_type);
    }

    return passed;
  }

private:
  bool run_verification(sycl::queue &queue, SrcT ref_value,
                        const std::string &src_data_type,
                        const std::string &dst_data_type) {
    shared_vector<DstT> result(NumElems, shared_allocator<DstT>(queue));
    shared_vector<SrcT> shared_ref_data(1, shared_allocator<SrcT>(queue));
    shared_ref_data.push_back(ref_value);

    queue.submit([&](sycl::handler &cgh) {
      const SrcT *const ref = shared_ref_data.data();
      DstT *const out = result.data();

      cgh.single_task<
          Kernel<SrcT, NumElems, DstT, TestCaseT, UsePositiveValueOnly>>(
          [=]() SYCL_ESIMD_KERNEL {
            TestCaseT::template call_simd_ctor<SrcT, DstT, NumElems>(ref[0],
                                                                     out);
          });
    });
    queue.wait_and_throw();

    const DstT expected = static_cast<DstT>(ref_value);
    bool passed = true;
    for (size_t i = 0; i < result.size(); ++i) {
      const DstT &retrieved = result[i];

      if constexpr (std::is_same_v<SrcT, DstT>) {
        if (!are_bitwise_equal(ref_value, retrieved)) {
          fail_test(i, retrieved, expected, ref_value, src_data_type,
                    dst_data_type);
        }
      } else if constexpr (type_traits::is_sycl_floating_point_v<DstT>) {
        // std::isnan() couldn't be called for integral types because it call is
        // ambiguous GitHub issue for that case:
        // https://github.com/microsoft/STL/issues/519
        if (!std::isnan(expected) && !std::isnan(retrieved)) {
          if (expected != retrieved) {
            passed = fail_test(i, retrieved, expected, ref_value, src_data_type,
                               dst_data_type);
          }
        }
      } else {
        if (expected != retrieved) {
          passed = fail_test(i, retrieved, expected, ref_value, src_data_type,
                             dst_data_type);
        }
      }
    }

    return passed;
  }

  bool fail_test(size_t index, DstT retrieved, DstT expected, SrcT ref_value,
                 const std::string &src_data_type,
                 const std::string &dst_data_type) {
    const auto description =
        BroadcastCtorTestDescription<SrcT, DstT, NumElems, TestCaseT>(
            index, retrieved, expected, ref_value, src_data_type,
            dst_data_type);
    log::fail(description);

    return false;
  }
};

template <typename SrcT, typename SizeT, typename DstT, typename TestCaseT>
using run_test_with_all_values =
    run_test<std::false_type, SrcT, SizeT, DstT, TestCaseT>;

template <typename SrcT, typename SizeT, typename DstT, typename TestCaseT>
using run_test_with_positive_value_only =
    run_test<std::true_type, SrcT, SizeT, DstT, TestCaseT>;

} // namespace esimd_test::api::functional::ctors
