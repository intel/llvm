// REQUIRES: cuda
// REQUIRES: windows

// RUN: %{build} -l d3d12 -l dxgi -l dxguid -o %t.out
// RUN: %t.out

#define TEST_SEMAPHORE_IMPORT
#include "read_write_unsampled.cpp"
