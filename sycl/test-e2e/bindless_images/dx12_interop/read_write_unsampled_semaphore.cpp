// REQUIRES: aspect-ext_oneapi_bindless_images
// REQUIRES: windows

// XFAIL: run-mode
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/15851

// DEFINE: %{link-flags}=%if cl_options %{ /clang:-ld3d12 /clang:-ldxgi /clang:-ldxguid %} %else %{ -ld3d12 -ldxgi -ldxguid %}
// RUN: %{build} %{link-flags} -o %t.out
// RUN: %{run-unfiltered-devices} %t.out

#define TEST_SEMAPHORE_IMPORT
// FIXME large image size fails in semaphore tests.
#define TEST_SMALL_IMAGE_SIZE
#include "read_write_unsampled.cpp"
