//==--------------- pipe_array.hpp - SYCL pipe array --------*- C++ -*------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include <CL/sycl/INTEL/pipes.hpp>
#include <utility>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace INTEL {

namespace {
template <size_t dim1, size_t... dims> struct VerifierDimLayer {
  template <size_t idx1, size_t... idxs> struct VerifierIdxLayer {
    static constexpr bool IsValid() {
      return idx1 < dim1 &&
             (VerifierDimLayer<dims...>::template VerifierIdxLayer<
                 idxs...>::IsValid());
    }
  };
};
template <size_t dim> struct VerifierDimLayer<dim> {
  template <size_t idx> struct VerifierIdxLayer {
    static constexpr bool IsValid() { return idx < dim; }
  };
};
} // namespace

template <class Id, typename BaseTy, size_t depth, size_t... dims>
struct PipeArray {
  PipeArray() = delete;

  template <size_t... idxs> struct StructId;

  template <size_t... idxs> struct VerifyIndices {
    static_assert(sizeof...(idxs) == sizeof...(dims),
                  "Indexing into a PipeArray requires as many indices as "
                  "dimensions of the PipeArray.");
    static_assert(VerifierDimLayer<dims...>::template VerifierIdxLayer<
                      idxs...>::IsValid(),
                  "Index out of bounds");
    using VerifiedPipe =
        cl::sycl::intel::pipe<StructId<idxs...>, BaseTy, depth>;
  };

  template <size_t... idxs>
  using PipeAt = typename VerifyIndices<idxs...>::VerifiedPipe;
};

} // namespace INTEL
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
