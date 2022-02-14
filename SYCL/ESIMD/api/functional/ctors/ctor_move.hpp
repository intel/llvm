//===-- ctor_move.hpp - Functions for tests on simd vector constructor
//      definition. -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides functions for tests on simd move constructor.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "common.hpp"

#include <algorithm>
#include <cassert>

namespace esimd = sycl::ext::intel::experimental::esimd;

namespace esimd_test::api::functional::ctors {

// Uses the initializer C++ context to call simd move constructor
struct initializer {
  static std::string get_description() { return "initializer"; }

  template <typename SimdT, typename ActionT>
  static void run(SimdT &&source, const ActionT &action) {
    static_assert(
        type_traits::is_nonconst_rvalue_reference_v<decltype(source)>);

    const auto instance = SimdT(std::move(source));
    action(instance);
  }
};

// Uses the variable declaration C++ context to call simd move constructor
struct var_decl {
  static std::string get_description() { return "variable declaration"; }

  template <typename SimdT, typename ActionT>
  static void run(SimdT &&source, const ActionT &action) {
    static_assert(
        type_traits::is_nonconst_rvalue_reference_v<decltype(source)>);

    const auto instance(std::move(source));
    action(instance);
  }
};

// Uses the rvalue in expression C++ context to call simd move constructor
struct rval_in_expr {
  static std::string get_description() { return "rvalue in expression"; }

  template <typename SimdT, typename ActionT>
  static void run(SimdT &&source, const ActionT &action) {
    static_assert(
        type_traits::is_nonconst_rvalue_reference_v<decltype(source)>);

    SimdT instance;
    instance = SimdT(std::move(source));
    action(instance);
  }
};

// Uses the function argument C++ context to call simd move constructor
class const_ref {
public:
  static std::string get_description() { return "const reference"; }

  template <typename SimdT, typename ActionT>
  static void run(SimdT &&source, const ActionT &action) {
    static_assert(
        type_traits::is_nonconst_rvalue_reference_v<decltype(source)>);

    action(SimdT(std::move(source)));
  }
};

// The core test functionality.
// Runs a TestCaseT, specific for each C++ context, for a simd<DataT,NumElems>
// instance
template <typename DataT, typename SizeT, typename TestCaseT> class run_test {
  static constexpr int NumElems = SizeT::value;
  using KernelName = Kernel<DataT, NumElems, TestCaseT>;

public:
  bool operator()(sycl::queue &queue, const std::string &data_type) {
    bool passed = true;
    bool was_moved = false;

    const shared_allocator<DataT> data_allocator(queue);
    const shared_allocator<int> flags_allocator(queue);
    const auto reference = generate_ref_data<DataT, NumElems>();

    shared_vector<DataT> input(reference.cbegin(), reference.cend(),
                               data_allocator);
    shared_vector<DataT> result(reference.size(), data_allocator);

    // We need a special handling for case of simd<T,1>, as we need to check
    // more than a single data value; therefore we need to loop over the
    // reference data to run test
    if constexpr (NumElems == 1) {
      const auto n_checks = input.size();
      const sycl::range<1> range(n_checks);

      // We need a separate flag per each check to have a parallel_for possible,
      // because any concurrent access to the same memory location is UB, even
      // in case we are updating variable to the same value in multiple threads
      shared_vector<int> flags_storage(n_checks, flags_allocator);

      // Run check for each of the reference elements using a single work-item
      // per single element
      queue.submit([&](sycl::handler &cgh) {
        const DataT *const ptr_in = input.data();
        const auto ptr_out = result.data();
        const auto ptr_flags = flags_storage.data();

        cgh.parallel_for<KernelName>(
            range, [=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
              const auto work_item_index = id[0];
              // Access a separate memory areas from each of the work items
              const DataT *const in = ptr_in + work_item_index;
              const auto out = ptr_out + work_item_index;
              const auto was_moved_flag = ptr_flags + work_item_index;
              *was_moved_flag = run_check(in, out);
            });
      });
      queue.wait_and_throw();

      // Oversafe: verify the proper signature was called for every check
      was_moved = std::all_of(flags_storage.cbegin(), flags_storage.cend(),
                              [](int flag) { return flag; });
    } else {
      assert((input.size() == NumElems) &&
             "Unexpected size of the input vector");

      shared_vector<int> flags_storage(1, flags_allocator);

      queue.submit([&](sycl::handler &cgh) {
        const DataT *const in = input.data();
        const auto out = result.data();
        const auto was_moved_flag = flags_storage.data();

        cgh.single_task<KernelName>(
            [=]() SYCL_ESIMD_KERNEL { *was_moved_flag = run_check(in, out); });
      });
      queue.wait_and_throw();

      was_moved = flags_storage[0];
    }

    if (!was_moved) {
      passed = false;

      // TODO: Make ITestDescription architecture more flexible
      std::string log_msg = "Failed for simd<";
      log_msg += data_type + ", " + std::to_string(NumElems) + ">";
      log_msg += ", with context: " + TestCaseT::get_description();
      log_msg += ". A copy constructor instead of a move constructor was used.";

      log::note(log_msg);
    } else {
      for (size_t i = 0; i < reference.size(); ++i) {
        const auto &retrieved = result[i];
        const auto &expected = reference[i];

        if (!are_bitwise_equal(retrieved, expected)) {
          passed = false;

          log::fail(ctors::TestDescription<DataT, NumElems, TestCaseT>(
              i, retrieved, expected, data_type));
        }
      }
    }

    return passed;
  }

private:
  // The core check logic.
  // Uses USM pointers for input data and to store the data from the new simd
  // instance, so that we could check it later
  // Returns the flag that should be true only if the move constructor was
  // actually called, to differentiate with the copy constructor calls
  static bool run_check(const DataT *const in, DataT *const out) {
    bool was_moved = false;

    // Prepare the source simd to move
    esimd::simd<DataT, NumElems> source;
    source.copy_from(in);

    // Action to run over the simd move constructor result
    const auto action = [&](const esimd::simd<DataT, NumElems> &instance) {
      was_moved = instance.get_test_proxy().was_move_destination();
      instance.copy_to(out);
    };
    // Call the move constructor in the specific context and run action
    // directly over the simd move constructor result
    TestCaseT::template run(std::move(source), action);

    return was_moved;
  }
};

} // namespace esimd_test::api::functional::ctors
