// REQUIRES: linux, gpu && level_zero
// REQUIRES: arch-intel_gpu_pvc
// ALLOW_RETRIES: 10

// RUN: %{run-aux} %{build} %device_tsan_aot_flags -O0 -g %S/Inputs/usm_data_race.cpp -o %t.out
// RUN: %{run} %t.out 2>&1 | FileCheck %S/Inputs/usm_data_race.cpp

// RUN: %{run-aux} %{build} %device_tsan_aot_flags -O1 -g %S/Inputs/usm_data_race.cpp -o %t.out
// RUN: %{run} %t.out 2>&1 | FileCheck %S/Inputs/usm_data_race.cpp

// RUN: %{run-aux} %{build} %device_tsan_aot_flags -O2 -g %S/Inputs/usm_data_race.cpp -o %t.out
// RUN: %{run} %t.out 2>&1 | FileCheck %S/Inputs/usm_data_race.cpp

// RUN: %{run-aux} %{build} %device_tsan_aot_flags -O3 -g %S/Inputs/usm_data_race.cpp -o %t.out
// RUN: %{run} %t.out 2>&1 | FileCheck %S/Inputs/usm_data_race.cpp
