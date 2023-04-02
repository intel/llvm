// REQUIRES: level_zero, level_zero_dev_kit
// RUN: env SYCL_BE=PI_LEVEL_ZERO %GPU_RUN_PLACEHOLDER %S/Inputs/interop-l0-direct-windows-old.out
// RUN: env SYCL_BE=PI_LEVEL_ZERO SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %GPU_RUN_PLACEHOLDER %S/interop-l0-direct-windows-old.out
// UNSUPPORTED: ze_debug-1,ze_debug4
// UNSUPPORTED: linux

#include "%S/Inputs/interop-l0-direct-old.cpp"
