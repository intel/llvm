// RUN: %{build} -Daccessor_new_api_test %S/Inputs/host_task_accessor.cpp -o %t.out
// RUN: %{run} %t.out

// Disabled on PVC without igc-dev due to timeout.
// UNSUPPORTED: arch-intel_gpu_pvc && !igc-dev
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/14826
