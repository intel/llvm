// RUN: %clangxx %fsycl-host-only -fsyntax-only -ferror-limit=0 -sycl-std=2020 -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

#include <sycl/sycl.hpp>

int main() {
  sycl::platform Plt;
  sycl::device Dev;

  // expected-warning@+1{{SYCL 2020 images are not supported on any devices. Consider using ‘aspect::ext_intel_legacy_image’ instead. Disable this warning with by defining SYCL_DISABLE_IMAGE_ASPECT_WARNING.}}
  std::ignore = Plt.has(sycl::aspect::image);

  // expected-warning@+1{{SYCL 2020 images are not supported on any devices. Consider using ‘aspect::ext_intel_legacy_image’ instead. Disable this warning with by defining SYCL_DISABLE_IMAGE_ASPECT_WARNING.}}
  std::ignore = Dev.has(sycl::aspect::image);

  return 0;
}
