//==-- CommandSubmitWrappers.hpp ---  -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/event.hpp>
#include <sycl/handler.hpp>

using namespace sycl;

template <typename KernelName, typename KernelType>
event single_task_wrapper(bool ShortcutSubmitFunction, queue &Q,
                          const KernelType &KernelFunc) {
  if (ShortcutSubmitFunction) {
    return Q.single_task<KernelName>(KernelFunc);
  } else {
    return Q.submit(
        [&](handler &cgh) { cgh.single_task<KernelName>(KernelFunc); });
  }
}

template <typename KernelName, typename KernelType>
event single_task_wrapper(bool ShortcutSubmitFunction, queue &Q, event DepEvent,
                          const KernelType &KernelFunc) {
  if (ShortcutSubmitFunction) {
    return Q.single_task<KernelName>(DepEvent, KernelFunc);
  } else {
    return Q.submit([&](handler &cgh) {
      cgh.depends_on(DepEvent);
      cgh.single_task<KernelName>(KernelFunc);
    });
  }
}