//===-- shared_elements.hpp - Function that provides USM with a smart pointer.
//      -------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides data struct that lets interact with USM with a smart
/// pointer that lets avoid memory leaks.
///
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

#include <functional>
#include <memory>

namespace esimd_test::api::functional {

// Provides APIs to interact with USM pointer without memory leaks for a single
// variable. Might be useful to provide access to a single boolean flag to store
// success, for example.
template <typename T> class shared_element {
  std::unique_ptr<T, std::function<void(T *)>> m_allocated_data;

public:
  shared_element(sycl::queue &queue, T initial_value) {
    const auto &device{queue.get_device()};
    const auto &context{queue.get_context()};

    auto deleter = [=](T *ptr) { sycl::free(ptr, context); };

    m_allocated_data = std::unique_ptr<T, decltype(deleter)>(
        sycl::malloc_shared<T>(1, device, context), deleter);

    assert(m_allocated_data && "USM memory allocation failed");
    *m_allocated_data = initial_value;
  }

  T *data() { return m_allocated_data.get(); }

  const T *data() const { return m_allocated_data.get(); }

  T value() { return *m_allocated_data.get(); }

  T value() const { return *m_allocated_data.get(); }
};

} // namespace esimd_test::api::functional
