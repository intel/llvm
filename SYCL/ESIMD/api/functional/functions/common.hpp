//===-- common.hpp - Define common code for simd functions tests ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides common code for simd functions tests.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "../common.hpp"

namespace esimd_test::api::functional::functions {

namespace esimd = sycl::ext::intel::esimd;

template <int NumElems, int NumSelectedElems, int Stride, int Offset,
          typename ContextT>
class TestDescription : public ITestDescription {
public:
  TestDescription(const std::string &data_type) : m_data_type(data_type) {}

  std::string to_string() const override {
    std::string test_description("simd<");

    test_description += m_data_type + ", " + std::to_string(NumElems) + ">";
    test_description += ", with context: " + ContextT::get_description();
    test_description +=
        ", with size selected elems: " + std::to_string(NumSelectedElems);
    test_description += ", with stride: " + std::to_string(Stride);
    test_description += ", with offset: " + std::to_string(Offset);

    return test_description;
  }

private:
  const std::string m_data_type;
};

} // namespace esimd_test::api::functional::functions
