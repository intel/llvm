//===-- ctor_load_acc.hpp - Generic code for tests on simd load constructors
//      from accessor -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides functions for tests on simd load constructor from
/// accessor
///
//===----------------------------------------------------------------------===//

#pragma once
#define ESIMD_TESTS_DISABLE_DEPRECATED_TEST_DESCRIPTION_FOR_LOGS

#include "../sycl_accessor.hpp"
#include "../sycl_range.hpp"
#include "ctor_load.hpp"

#include <cassert>
#include <cstring>
#include <memory>
#include <sstream>

namespace esimd_test::api::functional::ctors {

// Utility class for offset generation based on alignment and multiplicator
// given
// Offset should be a multiplier of alignment for simd load constructor to work
// as expected
template <unsigned int multiplicator> struct offset_generator {
  template <typename AlignmentT, typename DataT, int VecSize>
  static constexpr unsigned int get() {
    return multiplicator * AlignmentT::template get_size<DataT, VecSize>();
  }
};

// Descriptor class for constructor call within initializer context
struct initializer {
  static std::string get_description() { return "initializer"; }

  template <typename DataT, int NumElems, typename AccessorT,
            typename AlignmentT>
  static void call_simd_ctor(DataT *const out, const AccessorT &acc,
                             unsigned int offset, AlignmentT alignment) {
    const auto instance = esimd::simd<DataT, NumElems>(acc, offset, alignment);
    instance.copy_to(out);
  }
};

// Descriptor class for constructor call within variable declaration context
struct var_decl {
  static std::string get_description() { return "variable declaration"; }

  template <typename DataT, int NumElems, typename AccessorT,
            typename AlignmentT>
  static void call_simd_ctor(DataT *const out, const AccessorT &acc,
                             unsigned int offset, AlignmentT alignment) {
    esimd::simd<DataT, NumElems> instance(acc, offset, alignment);
    instance.copy_to(out);
  }
};

// Descriptor class for constructor call within r-value in an expression context
struct rval_in_expr {
  static std::string get_description() { return "rvalue in an expression"; }

  template <typename DataT, int NumElems, typename AccessorT,
            typename AlignmentT>
  static void call_simd_ctor(DataT *const out, const AccessorT &acc,
                             unsigned int offset, AlignmentT alignment) {
    esimd::simd<DataT, NumElems> instance;
    instance = esimd::simd<DataT, NumElems>(acc, offset, alignment);
    instance.copy_to(out);
  }
};

// Descriptor class for constructor call within const reference context
class const_ref {
public:
  static std::string get_description() { return "const reference"; }

  template <typename DataT, int NumElems, typename AccessorT,
            typename AlignmentT>
  static void call_simd_ctor(DataT *const out, const AccessorT &acc,
                             unsigned int offset, AlignmentT alignment) {
    call_simd_by_const_ref<DataT, NumElems>(
        esimd::simd<DataT, NumElems>(acc, offset, alignment), out);
  }

private:
  template <typename DataT, int NumElems>
  static void
  call_simd_by_const_ref(const esimd::simd<DataT, NumElems> &simd_by_const_ref,
                         DataT *const out) {
    simd_by_const_ref.copy_to(out);
  }
};

// Test case description for logging purposes
template <int VecSize, typename ContextT>
struct LoadCtorAccTestDescription : public ITestDescription {
  using BaseT = ctors::LoadCtorTestDescription<VecSize, ContextT>;

public:
  LoadCtorAccTestDescription(const std::string &vec_data_name, int dims,
                             const std::string &acc_mode_name,
                             const std::string &acc_target_name,
                             unsigned int offset,
                             const std::string &alignment_name) {
    std::ostringstream stream;

    using BaseDescriptionT = ctors::LoadCtorTestDescription<VecSize, ContextT>;
    const BaseDescriptionT base_description(vec_data_name, alignment_name);
    const AccessorDescription accessor_description(
        vec_data_name, dims, acc_mode_name, acc_target_name);

    stream << base_description.to_string();
    stream << ", with offset " << log::stringify(offset);
    stream << ", from " << accessor_description.to_string();

    m_description = stream.str();
  }

  std::string to_string() const override { return m_description; }

private:
  std::string m_description;
};

// The main test routine
//
template <typename DataT, typename VecSizeT, typename AccDimsT,
          typename AccModeT, typename AccTargetT, typename ContextT,
          typename OffsetGeneratorT, typename AlignmentT>
class run_test {
  static constexpr int VecSize = VecSizeT::value;
  static constexpr int AccDims = AccDimsT::value;
  static constexpr sycl::access_mode AccMode = AccModeT::value;
  static constexpr sycl::target AccTarget = AccTargetT::value;
  static constexpr auto Offset =
      OffsetGeneratorT::template get<AlignmentT, DataT, VecSize>();

  using SimdT = esimd::simd<DataT, VecSize>;
  using AccessorT = sycl::accessor<DataT, AccDims, AccMode, AccTarget>;
  using TestDescriptionT = LoadCtorAccTestDescription<VecSize, ContextT>;
  using KernelT = Kernel<DataT, VecSize, AccDimsT, AccModeT, AccTargetT,
                         ContextT, OffsetGeneratorT, AlignmentT>;

  static_assert(AccTarget == sycl::target::device,
                "Accessor target is not supported");

public:
  bool operator()(sycl::queue &queue, const std::string &vec_data_name,
                  const std::string &acc_mode_name,
                  const std::string &acc_target_name,
                  const std::string &alignment_name) {
    // Define the mapping between parameters retrieved and test descriptor
    // arguments for logging purposes
    return run(queue, /* The rest are the test description parameters */
               vec_data_name, AccDims, acc_mode_name, acc_target_name, Offset,
               alignment_name);
  }

private:
  template <typename... TestDescriptionArgsT>
  inline bool run(sycl::queue &queue, TestDescriptionArgsT &&...args) {
    bool passed = true;
    log::trace<TestDescriptionT>(std::forward<TestDescriptionArgsT>(args)...);

    if (should_skip_test_with<DataT>(queue.get_device()))
      return passed;

    const std::vector<DataT> ref_data = generate_ref_data<DataT, VecSize>();

    if constexpr (VecSize == 1) {
      // Ensure simd load constructor works as expected with every value from
      // reference data
      for (size_t i = 0; i < ref_data.size(); ++i) {
        passed = run_with_data(queue, {ref_data[i]},
                               std::forward<TestDescriptionArgsT>(args)...);
      }
    } else {
      passed = run_with_data(queue, ref_data,
                             std::forward<TestDescriptionArgsT>(args)...);
    }
    return passed;
  }

  template <typename... TestDescriptionArgsT>
  bool run_with_data(sycl::queue &queue, const std::vector<DataT> &ref_data,
                     TestDescriptionArgsT... args) {
    assert(ref_data.size() == VecSize &&
           "Reference data size is not equal to the simd vector length.");

    bool passed = true;
    std::vector<DataT> container;

    shared_allocator<DataT> allocator(queue);
    shared_vector<DataT> result(VecSize, allocator);

    // Fill container with reference data using the pointer modified accordingly
    // to the offset parameter of the load constructor
    {
      const size_t extra_space =
          (Offset / sizeof(DataT)) + (Offset % sizeof(DataT) > 0);
      container.resize(VecSize + extra_space);
      log::debug([&]() {
        return " ... using container with " + log::stringify(container.size()) +
               " elements";
      });

      // We don't break the strict aliasing rule according to the C++17
      // [basic.lval]
      auto ptr = reinterpret_cast<unsigned char *>(container.data());
      std::memset(ptr, 0, Offset);
      ptr += Offset;
      std::memcpy(ptr, ref_data.data(), VecSize * sizeof(DataT));
      // No initialization for the rest of the bytes to simplify the code

      // Now container has the reference elements starting from the offset byte,
      // with all preceding bytes filled by zero.
      // No other modification required, as the simd load constructor doesn't
      // actually work with alignment of input data
    }

    // Call simd constructor and fill the result vector
    {
      const auto range = get_sycl_range<AccDims>(container.size());
      sycl::buffer<DataT, AccDims> buffer(container.data(), range);
      log::debug([&]() {
        return " ... using sycl::buffer with " + log::stringify(range) +
               " to access container";
      });
      assert((container.size() == range.size()) && "Unexpected range");

      queue.submit([&](sycl::handler &cgh) {
        const AccessorT acc =
            buffer.template get_access<AccMode, AccTarget>(cgh);
        DataT *const out = result.data();

        cgh.single_task<KernelT>([=]() SYCL_ESIMD_KERNEL {
          // This alignment affect only the internal simd storage
          // efficiency; no any test failure expected with any alignment
          // provided for every possible case
          const auto alignment = AlignmentT::get_value();
          ContextT::template call_simd_ctor<DataT, VecSize>(out, acc, Offset,
                                                            alignment);
        });
      });
      queue.wait_and_throw();
    }

    // Validate results
    for (size_t i = 0; i < result.size(); ++i) {
      const auto &expected = ref_data[i];
      const auto &retrieved = result[i];

      if (!are_bitwise_equal(expected, retrieved)) {
        passed = false;

        log::fail(TestDescriptionT(std::forward<TestDescriptionArgsT>(args)...),
                  "Unexpected value at index ", i, ", retrieved: ", retrieved,
                  ", expected: ", expected);
      }
    }
    return passed;
  }
};

} // namespace esimd_test::api::functional::ctors
