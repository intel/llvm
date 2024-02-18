//==----- preview_marker.cpp --- Preview library marker symbol -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/export.hpp>

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
namespace sycl {
inline namespace _V1 {
namespace detail {

// Exported marker function to help verify that the preview library correctly
// defines the __INTEL_PREVIEW_BREAKING_CHANGES macro and is linked with when
// the -fpreview-breaking-changes option is used.
__SYCL_EXPORT void PreviewMajorReleaseMarker() {}

} // namespace detail
} // namespace _V1
} // namespace sycl
#endif // __INTEL_PREVIEW_BREAKING_CHANGES
