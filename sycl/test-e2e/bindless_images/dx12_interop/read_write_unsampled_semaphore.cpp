// REQUIRES: cuda
// REQUIRES: windows

// RUN: %{build} -l d3d12 -l dxgi -l dxguid -o %t.out
// RUN: %t.out

#define TEST_SEMAPHORE_IMPORT
// FIXME large image size fails in semaphore tests.
#define TEST_SMALL_IMAGE_SIZE
#include "read_write_unsampled.cpp"
