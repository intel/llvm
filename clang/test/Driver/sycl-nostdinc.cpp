// RUN: %clangxx -fsycl -fsycl-device-only -nostdlibinc -fsyntax-only %s
// RUN: %clangxx -fsycl -fsycl-device-only -nostdinc -fsyntax-only %s

// RUN: %clangxx -fsycl -nostdlibinc -fsyntax-only %s
// RUN: %clangxx -fsycl -nostdinc -fsyntax-only %s

#if __has_include(<sycl/sycl.hpp>)
#error "expected to *not* be able to find SYCL headers"
#endif
