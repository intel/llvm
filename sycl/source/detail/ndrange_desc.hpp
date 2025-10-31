//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/nd_range.hpp>
#include <sycl/range.hpp>

#include <array>

namespace sycl {
inline namespace _V1 {
namespace detail {
// The structure represents NDRange - global, local sizes, global offset and
// number of dimensions.

// TODO: A lot of tests rely on particular values to be set for dimensions that
// are not used. To clarify, for example, if a 2D kernel is invoked, in
// NDRDescT, the value of index 2 in GlobalSize must be set to either 1 or 0
// depending on which constructor is used for no clear reason.
// Instead, only sensible defaults should be used and tests should be updated
// to reflect this.
class NDRDescT {

public:
  NDRDescT() = default;
  NDRDescT(const NDRDescT &Desc) = default;
  NDRDescT(NDRDescT &&Desc) = default;

  NDRDescT(const size_t *N, bool SetNumWorkGroups, int DimsVal)
      : Dims{size_t(DimsVal)} {
    if (SetNumWorkGroups) {
      for (size_t I = 0; I < Dims; ++I) {
        NumWorkGroups[I] = N[I];
      }
    } else {
      for (size_t I = 0; I < Dims; ++I) {
        GlobalSize[I] = N[I];
      }

      for (int I = Dims; I < 3; ++I) {
        GlobalSize[I] = 1;
      }
    }
  }

  template <int Dims_>
  NDRDescT(sycl::range<Dims_> N, bool SetNumWorkGroups)
      : NDRDescT(&(N[0]), SetNumWorkGroups, Dims_) {}

  NDRDescT(const size_t *NumWorkItems, const size_t *LocalSizes,
           const size_t *Offset, int DimsVal)
      : Dims{size_t(DimsVal)} {
    for (size_t I = 0; I < Dims; ++I) {
      GlobalSize[I] = NumWorkItems[I];
      LocalSize[I] = LocalSizes[I];
      GlobalOffset[I] = Offset[I];
    }

    for (int I = Dims; I < 3; ++I) {
      LocalSize[I] = LocalSizes[0] ? 1 : 0;
    }

    for (int I = Dims; I < 3; ++I) {
      GlobalSize[I] = 1;
    }
  }

  template <int Dims_>
  NDRDescT(sycl::range<Dims_> NumWorkItems, sycl::range<Dims_> LocalSizes,
           sycl::id<Dims_> Offset)
      : NDRDescT(&(NumWorkItems[0]), &(LocalSizes[0]), &(Offset[0]), Dims_) {}

  NDRDescT(const size_t *NumWorkItems, const size_t *Offset, int DimsVal)
      : Dims{size_t(DimsVal)} {
    for (size_t I = 0; I < Dims; ++I) {
      GlobalSize[I] = NumWorkItems[I];
      GlobalOffset[I] = Offset[I];
    }
  }

  template <int Dims_>
  NDRDescT(sycl::range<Dims_> NumWorkItems, sycl::id<Dims_> Offset)
      : NDRDescT(&(NumWorkItems[0]), &(Offset[0]), Dims_) {}

  template <int Dims_>
  NDRDescT(sycl::nd_range<Dims_> ExecutionRange)
      : NDRDescT(ExecutionRange.get_global_range(),
                 ExecutionRange.get_local_range(),
                 ExecutionRange.get_offset()) {}

  template <int Dims_>
  NDRDescT(sycl::range<Dims_> Range)
      : NDRDescT(Range, /*SetNumWorkGroups=*/false) {}

  template <int Dims_> void setClusterDimensions(sycl::range<Dims_> N) {
    if (this->Dims != size_t(Dims_)) {
      throw std::runtime_error(
          "Dimensionality of cluster, global and local ranges must be same");
    }

    for (int I = 0; I < Dims_; ++I)
      ClusterDimensions[I] = N[I];
  }

  NDRDescT &operator=(const NDRDescT &Desc) = default;
  NDRDescT &operator=(NDRDescT &&Desc) = default;

  std::array<size_t, 3> GlobalSize{0, 0, 0};
  std::array<size_t, 3> LocalSize{0, 0, 0};
  std::array<size_t, 3> GlobalOffset{0, 0, 0};
  /// Number of workgroups, used to record the number of workgroups from the
  /// simplest form of parallel_for_work_group. If set, all other fields must be
  /// zero
  std::array<size_t, 3> NumWorkGroups{0, 0, 0};
  std::array<size_t, 3> ClusterDimensions{1, 1, 1};
  size_t Dims = 0;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
