//===-- ctor_load.hpp - Functions for tests on simd load constructor.
//      -------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides functions for tests on simd load constructor
///
//===----------------------------------------------------------------------===//

#pragma once

#include "common.hpp"

#include <string>

namespace esimd = sycl::ext::intel::esimd;

namespace esimd_test::api::functional::ctors {

// Dummy kernel for submitting some code into device side.
template <typename DataT, int NumElems, typename T, typename Alignment>
struct Kernel_for_load_ctor;

// Alignment tags
namespace alignment {

struct element {
  static std::string to_string() { return "element_aligned"; }
  template <typename DataT, int> static constexpr size_t get_size() {
    return alignof(DataT);
  }
  static constexpr auto get_value() { return esimd::element_aligned; }
};

struct vector {
  static std::string to_string() { return "vector_aligned"; }
  template <typename DataT, int NumElems> static constexpr size_t get_size() {
    // Referring to the simd class specialization on the host side is by design.
    return alignof(esimd::simd<DataT, NumElems>);
  }
  static constexpr auto get_value() { return esimd::vector_aligned; }
};

template <unsigned int size = 16 /*oword alignment*/> struct overal {
  // Use 16 instead of std::max_align_t because of the fact that long double is
  // not a native type in Intel GPUs. So 16 is not driven by any type, but
  // rather the "oword alignment" requirement for all block loads. In that
  // sense, std::max_align_t would give wrong idea.

  static std::string to_string() {
    return "overaligned<" + std::to_string(size) + ">";
  }

  template <typename DataT, int> static constexpr size_t get_size() {
    static_assert(size % alignof(DataT) == 0,
                  "Unsupported data type alignment");
    return size;
  }

  static constexpr auto get_value() { return esimd::overaligned<size>; }
};

} // namespace alignment

// Detailed test case description to use for logs
template <int NumElems, typename TestCaseT>
class LoadCtorTestDescription : public ITestDescription {
public:
  LoadCtorTestDescription(const std::string &data_type,
                          const std::string &alignment_name)
      : m_description(data_type), m_alignment_name(alignment_name) {}

  std::string to_string() const override {
    return m_description.to_string() + ", with alignment: " + m_alignment_name;
  }

private:
  const ctors::TestDescription<NumElems, TestCaseT> m_description;
  const std::string m_alignment_name;
};

} // namespace esimd_test::api::functional::ctors
