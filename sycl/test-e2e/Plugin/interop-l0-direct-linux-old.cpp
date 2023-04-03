// REQUIRES: level_zero
// RUN: env SYCL_BE=PI_LEVEL_ZERO %GPU_RUN_PLACEHOLDER %S/Inputs/interop-l0-direct-linux-old.out
// RUN: env SYCL_BE=PI_LEVEL_ZERO SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %GPU_RUN_PLACEHOLDER %S/Inputs/interop-l0-direct-linux-old.out
// UNSUPPORTED: ze_debug-1,ze_debug4
// UNSUPPORTED: windows

#include "%S/Inputs/interop-l0-direct-old.cpp"
