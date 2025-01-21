//===-- common.hpp - Define common code for simd operators tests ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides common things for simd operators tests.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "../common.hpp"
#include <string>

namespace esimd_test::api::functional::operators {

#ifdef ESIMD_TESTS_DISABLE_DEPRECATED_TEST_DESCRIPTION_FOR_LOGS

template <int NumElems, typename TestCaseT>
class TestDescription : public ITestDescription {
public:
  TestDescription(const std::string &data_type) : m_data_type(data_type) {}

  std::string to_string() const override {
    std::string test_description = TestCaseT::get_description();

    test_description += " with simd<" + m_data_type;
    test_description += ", " + std::to_string(NumElems) + ">";

    return test_description;
  }

private:
  const std::string m_data_type;
};

#else

// Deprecated, use TestDescription<NumElems, ContextT> for new tests instead
//
// TODO: Remove deprecated TestDescription from all tests
template <typename DataT, int NumElems>
class [[deprecated]]	TestDescription : public ITestDescription {
public:
  TestDescription(size_t index, DataT retrieved_val, DataT expected_val,
                  const std::string &data_type)
      : m_data_type(data_type), m_retrieved_val(retrieved_val),
        m_expected_val(expected_val), m_index(index) {}

  std::string to_string() const override {
    std::string log_msg("Failed for simd<");

    log_msg += m_data_type + ", " + std::to_string(NumElems) + ">";
    log_msg += ", retrieved: " + std::to_string(m_retrieved_val);
    log_msg += ", expected: " + std::to_string(m_expected_val);
    log_msg += ", at index: " + std::to_string(m_index);

    return log_msg;
  }

private:
  const std::string m_data_type;
  const DataT m_retrieved_val;
  const DataT m_expected_val;
  const size_t m_index;
};

#endif // ESIMD_TESTS_DISABLE_DEPRECATED_TEST_DESCRIPTION_FOR_LOGS

} // namespace esimd_test::api::functional::operators
