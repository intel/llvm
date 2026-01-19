//==-- CommandSubmitWrappers.hpp ----- Wrappers for command submission -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/event.hpp>
#include <sycl/handler.hpp>
#include <sycl/queue.hpp>

namespace sycl {

inline namespace _V1 {
namespace unittest {

// Wrappers introduced in this file allow for running unit tests
// with two command submission types: Using a handler and handler-less
// shortcut functions.
// This increases the test coverage, especially for the cases,
// where the command submission path implementation differes significantly
// between those two models.

template <typename KernelName, typename KernelType>
event single_task_wrapper(bool UseShortcutFunction, queue &Q,
                          const KernelType &KernelFunc) {
  if (UseShortcutFunction) {
    return Q.single_task<KernelName>(KernelFunc);
  } else {
    return Q.submit(
        [&](handler &cgh) { cgh.single_task<KernelName>(KernelFunc); });
  }
}

template <typename KernelName, typename KernelType>
event single_task_wrapper(bool UseShortcutFunction, queue &Q, event &DepEvent,
                          const KernelType &KernelFunc) {
  if (UseShortcutFunction) {
    return Q.single_task<KernelName>(DepEvent, KernelFunc);
  } else {
    return Q.submit([&](handler &cgh) {
      cgh.depends_on(DepEvent);
      cgh.single_task<KernelName>(KernelFunc);
    });
  }
}
} // namespace unittest
} // namespace _V1
} // namespace sycl
