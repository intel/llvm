//==--------- compute_units.hpp - SYCL Compute Units -------*- C++ -*-------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl.hpp>
#include <CL/sycl/detail/defines.hpp>
#include <utility>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace INTEL {

namespace {
template <class Func, template <std::size_t> class Name, std::size_t index>
class SubmitOneComputeUnit {
public:
  SubmitOneComputeUnit(Func &&f, sycl::queue &q) {

    q.submit([&](sycl::handler &h) {
      h.single_task<Name<index>>([=]() {
        // verifies that f only takes a single argument
        f(std::integral_constant<std::size_t, index>());
      });
    });
  }
};

template <template <std::size_t> class Name, class Func, std::size_t... index>
inline constexpr void ComputeUnitUnroller(sycl::queue &q, Func &&f,
                                          std::index_sequence<index...>) {
  (SubmitOneComputeUnit<Func, Name, index>(f, q), ...); // fold expression
}
} // namespace

// N is the number of compute units
// Name is the kernel's name
template <std::size_t N, template <std::size_t ID> class Name, class Func>
constexpr void submit_compute_units(sycl::queue &q, Func &&f) {
  std::make_index_sequence<N> indices;
  ComputeUnitUnroller<Name>(q, f, indices);
}

} // namespace INTEL
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
