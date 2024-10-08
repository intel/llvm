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
#define ESIMD_TESTS_DISABLE_DEPRECATED_TEST_DESCRIPTION_FOR_LOGS

#include "common.hpp"
#include <cassert>
// For std::isnan
#include <cmath>

namespace esimd = sycl::ext::intel::esimd;
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

// Class used as a kernel ID.
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

inline std::string init_val_to_string(init_val val) {
  switch (val) {
  case init_val::min:
    return "lowest";
    break;
  case init_val::max:
    return "max";
    break;
  case init_val::zero:
    return "zero";
    break;
  case init_val::positive:
    return "positive";
    break;
  case init_val::negative:
    return "negative";
    break;
  case init_val::min_half:
    return "min_half";
    break;
  case init_val::max_half:
    return "max_half";
    break;
  case init_val::neg_inf:
    return "neg_inf";
    break;
  case init_val::nan:
    return "nan";
    break;
  case init_val::denorm:
    return "denorm";
    break;
  case init_val::inexact:
    return "inexact";
    break;
  case init_val::ulp:
    return "ulp";
    break;
  case init_val::ulp_half:
    return "ulp_half";
    break;
  default:
    assert(false && "Unexpected enum value");
  };
  return "n/a";
}

template <int NumElems, typename ContextT>
class FillCtorTestDescription : public ITestDescription {
public:
  FillCtorTestDescription(const std::string &data_type, init_val base_val,
                          init_val step)
      : m_description(data_type), m_base_val(base_val), m_step(step) {}

  std::string to_string() const override {
    std::string log_msg = m_description.to_string();

    log_msg += ", with base value: " + init_val_to_string(m_base_val);
    log_msg += ", with step value: " + init_val_to_string(m_step);

    return log_msg;
  }

private:
  const ctors::TestDescription<NumElems, ContextT> m_description;
  const init_val m_base_val;
  const init_val m_step;
};

template <init_val... Values> auto get_init_values_pack() {
  return value_pack<init_val, Values...>::generate_unnamed();
}

template <typename DataT, typename SizeT, typename TestCaseT, typename BaseValT,
          typename StepT>
class run_test {
  static constexpr int NumElems = SizeT::value;
  static constexpr init_val BaseVal = BaseValT::value;
  static constexpr init_val Step = StepT::value;
  using KernelT = kernel_for_fill<DataT, NumElems, TestCaseT, BaseVal, Step>;
  using TestDescriptionT = FillCtorTestDescription<NumElems, TestCaseT>;

public:
  bool operator()(sycl::queue &queue, const std::string &data_type) {
    bool passed = true;
    log::trace<TestDescriptionT>(data_type, BaseVal, Step);

    if (should_skip_test_with<DataT>(queue.get_device())) {
      return true;
    }

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

    // Verify the the fill constructor.
    for (size_t i = 0; i < result.size(); ++i) {
      if constexpr (BaseVal == init_val::nan || Step == init_val::nan) {

        if (!std::isnan(result[i])) {
          passed = false;
          log::fail(TestDescriptionT(data_type, BaseVal, Step),
                    "Unexpected value at index ", i, ", retrieved: ", result[i],
                    ", expected: any NaN value");
        }
      } else {

        DataT expected_value = base_value + (DataT)i * step_value;
        if (!are_bitwise_equal(result[i], expected_value)) {
          passed = false;
          log::fail(TestDescriptionT(data_type, BaseVal, Step),
                    "Unexpected value at index ", i, ", retrieved: ", result[i],
                    ", expected: ", expected_value);
        }
      }
    }
    return passed;
  }
};

} // namespace esimd_test::api::functional::ctors
