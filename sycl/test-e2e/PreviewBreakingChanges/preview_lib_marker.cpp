// REQUIRES: preview-breaking-changes-supported

// RUN: %clangxx -fsycl -fpreview-breaking-changes %s -o %t.out
// RUN: %{run} %t.out

// Test to help identify that E2E testing correctly detects and uses the preview
// library.

#include <sycl/detail/core.hpp>

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
