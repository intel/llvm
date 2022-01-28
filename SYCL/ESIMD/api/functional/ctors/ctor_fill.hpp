//===-- ctor_fill.hpp - Functions for tests on simd fill constructor
//      definition. -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides functions for tests on simd fill constructor.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "common.hpp"
// For std::isnan
#include <cmath>

namespace esimd = sycl::ext::intel::experimental::esimd;
namespace esimd_functional = esimd_test::api::functional;

namespace esimd_test::api::functional::ctors {

// Descriptor class for the case of calling constructor in initializer context.
struct initializer {
  static std::string get_description() { return "initializer"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(DataT init_value, DataT step, DataT *const out) {
    const auto simd_by_init = esimd::simd<DataT, NumElems>(init_value, step);
    simd_by_init.copy_to(out);
  }
};

// Descriptor class for the case of calling constructor in variable declaration
// context.
struct var_decl {
  static std::string get_description() { return "variable declaration"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(DataT init_value, DataT step, DataT *const out) {
    const esimd::simd<DataT, NumElems> simd_by_var_decl(init_value, step);
    simd_by_var_decl.copy_to(out);
  }
};

// Descriptor class for the case of calling constructor in rvalue in an
// expression context.
struct rval_in_expr {
  static std::string get_description() { return "rvalue in an expression"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(DataT init_value, DataT step, DataT *const out) {
    esimd::simd<DataT, NumElems> simd_by_rval;
    simd_by_rval = esimd::simd<DataT, NumElems>(init_value, step);
    simd_by_rval.copy_to(out);
  }
};

// Descriptor class for the case of calling constructor in const reference
// context.
class const_ref {
public:
  static std::string get_description() { return "const reference"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(DataT init_value, DataT step, DataT *const out) {
    return call_simd_by_const_ref<DataT, NumElems>(
        esimd::simd<DataT, NumElems>(init_value, step), out);
  }

private:
  template <typename DataT, int NumElems>
  static void
  call_simd_by_const_ref(const esimd::simd<DataT, NumElems> &simd_by_const_ref,
                         DataT *const out) {
    simd_by_const_ref.copy_to(out);
  }
};

// Enumeration of possible values for base value and step that will be provided
// into simd constructor.
enum class init_val {
  min,
  max,
  zero,
  min_half,
  max_half,
  neg_inf,
  nan,
  positive,
  negative,
  denorm,
  inexact,
  ulp,
  ulp_half
};

// Dummy kernel for submitting some code into device side.
template <typename DataT, int NumElems, typename T, init_val BaseVal,
          init_val StepVal>
struct kernel_for_fill;

// Constructing a value for step and base values that depends on input
// parameters.
template <typename DataT, init_val Value>
DataT get_value(DataT base_val = DataT()) {
  if constexpr (Value == init_val::min) {
    return value<DataT>::lowest();
  } else if constexpr (Value == init_val::max) {
    return value<DataT>::max();
  } else if constexpr (Value == init_val::zero) {
    return 0;
  } else if constexpr (Value == init_val::positive) {
    return static_cast<DataT>(1.25);
  } else if constexpr (Value == init_val::negative) {
    return static_cast<DataT>(-10.75);
  } else if constexpr (Value == init_val::min_half) {
    return value<DataT>::lowest() / 2;
  } else if constexpr (Value == init_val::max_half) {
    return value<DataT>::max() / 2;
  } else if constexpr (Value == init_val::neg_inf) {
    return -value<DataT>::inf();
  } else if constexpr (Value == init_val::nan) {
    return value<DataT>::nan();
  } else if constexpr (Value == init_val::denorm) {
    return value<DataT>::denorm_min();
  } else if constexpr (Value == init_val::inexact) {
    return 0.1;
  } else if constexpr (Value == init_val::ulp) {
    return value<DataT>::pos_ulp(base_val);
  } else if constexpr (Value == init_val::ulp_half) {
    return value<DataT>::pos_ulp(base_val) / 2;
  } else {
    static_assert(Value != Value, "Unexpected enum value");
  }
}

template <init_val Val> std::string init_val_to_string() {
  if constexpr (Val == init_val::min) {
    return "lowest";
  } else if constexpr (Val == init_val::max) {
    return "max";
  } else if constexpr (Val == init_val::zero) {
    return "zero";
  } else if constexpr (Val == init_val::positive) {
    return "positive";
  } else if constexpr (Val == init_val::negative) {
    return "negative";
  } else if constexpr (Val == init_val::min_half) {
    return "min_half";
  } else if constexpr (Val == init_val::max_half) {
    return "max_half";
  } else if constexpr (Val == init_val::neg_inf) {
    return "neg_inf";
  } else if constexpr (Val == init_val::nan) {
    return "nan";
  } else if constexpr (Val == init_val::denorm) {
    return "denorm";
  } else if constexpr (Val == init_val::inexact) {
    return "inexact";
  } else if constexpr (Val == init_val::ulp) {
    return "ulp";
  } else if constexpr (Val == init_val::ulp_half) {
    return "ulp_half";
  } else {
    static_assert(Val != Val, "Unexpected enum value");
  }
}

template <typename DataT, int NumElems, typename ContextT, init_val BaseVal,
          init_val Step>
class FillCtorTestDescription
    : public TestDescription<DataT, NumElems, ContextT> {
public:
  FillCtorTestDescription(size_t index, DataT retrieved_val, DataT expected_val,
                          const std::string &data_type)
      : TestDescription<DataT, NumElems, ContextT>(index, retrieved_val,
                                                   expected_val, data_type) {}

  std::string to_string() const override {
    std::string log_msg(
        TestDescription<DataT, NumElems, ContextT>::to_string());

    log_msg += ", with base value: " + init_val_to_string<BaseVal>();
    log_msg += ", with step value: " + init_val_to_string<Step>();

    return log_msg;
  }
};

template <init_val... Values> auto get_init_values_pack() {
  return value_pack<init_val, Values...>::generate_unnamed();
}

template <typename DataT, typename DimT, typename TestCaseT, typename BaseValT,
          typename StepT>
class run_test {
  static constexpr int NumElems = DimT::value;
  static constexpr init_val BaseVal = BaseValT::value;
  static constexpr init_val Step = StepT::value;
  using KernelT = kernel_for_fill<DataT, NumElems, TestCaseT, BaseVal, Step>;

public:
  bool operator()(sycl::queue &queue, const std::string &data_type) {
    shared_vector<DataT> result(NumElems, shared_allocator<DataT>(queue));

    const auto base_value = get_value<DataT, BaseVal>();
    const auto step_value = get_value<DataT, Step>(base_value);

    queue.submit([&](sycl::handler &cgh) {
      DataT *const out = result.data();

      cgh.single_task<KernelT>([=]() SYCL_ESIMD_KERNEL {
        TestCaseT::template call_simd_ctor<DataT, NumElems>(base_value,
                                                            step_value, out);
      });
    });
    queue.wait_and_throw();
    bool passed = true;

    // Verify the base value was passed as-is
    if (!are_bitwise_equal(result[0], base_value)) {
      passed = fail_test(0, result[0], base_value, data_type);
    }

    // Verify the step value works as expected being passed to the fill
    // constructor.
    DataT expected_value = base_value;
    for (size_t i = 1; i < result.size(); ++i) {
      if constexpr (BaseVal == init_val::nan || Step == init_val::nan) {

        if (!std::isnan(result[i])) {
          passed = false;

          // TODO: Make ITestDescription architecture more flexible.
          // We are assuming that the NaN opcode may differ
          std::string log_msg = "Failed for simd<";
          log_msg += data_type + ", " + std::to_string(NumElems) + ">";
          log_msg += ", with context: " + TestCaseT::get_description();
          log_msg += ". The element at index: " + std::to_string(i) +
                     ", is not nan, but it should.";
          log_msg += ", with base value: " + init_val_to_string<BaseVal>();
          log_msg += ", with step value: " + init_val_to_string<Step>();

          log::note(log_msg);
        }
      } else {

        expected_value += step_value;
        if (!are_bitwise_equal(result[i], expected_value)) {
          passed = fail_test(i, result[i], expected_value, data_type);
        }
      }
    }
    return passed;
  }

private:
  bool fail_test(size_t index, DataT retrieved, DataT expected,
                 const std::string &data_type) {
    const auto description =
        FillCtorTestDescription<DataT, NumElems, TestCaseT, BaseVal, Step>(
            index, retrieved, expected, data_type);
    log::fail(description);

    return false;
  }
};

} // namespace esimd_test::api::functional::ctors
