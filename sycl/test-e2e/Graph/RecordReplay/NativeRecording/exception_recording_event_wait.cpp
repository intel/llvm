// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21
// REQUIRES: linux
// REQUIRES-INTEL-DRIVER: lin: 38583

// TODO: Add minimum Windows driver version once available

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

#define GRAPH_E2E_NATIVE_RECORDING

#include "../../Inputs/exception_recording_event_wait.cpp"
