// RUN: not %clangxx -fsycl %s -o %t 2>&1 | FileCheck --check-prefix=CHECK-NO-PREVIEW %s
// RUN: %clangxx -fsycl -fpreview-breaking-changes %s -o %t
// REQUIRES: preview-breaking-changes-lib && windows

// Checks that the preview-breaking-changes marker is present only when the
// -fpreview-breaking-changes option is used. This implies two things:
//  1. The driver links against the right library, i.e. sycl-preview.
//  2. The sycl-preview library has the __INTEL_PREVIEW_BREAKING_CHANGES macro
//     defined.

#include <sycl/sycl.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

extern void PreviewMajorReleaseMarker();

} // namespace detail
} // namespace _V1
} // namespace sycl

int main() {
  sycl::detail::PreviewMajorReleaseMarker();
  return 0;
}

// CHECK-NO-PREVIEW: unresolved external symbol "void __cdecl sycl::_V1::detail::PreviewMajorReleaseMarker(void)"
