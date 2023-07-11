// RUN: %clangxx -fsycl -fsycl-targets="native_cpu,spir64" -### %s 2>&1 | FileCheck %s

// checks that we emit the correct warning when native_cpu is listed together with other sycl targets
// TODO: remove this test and the warning once native_cpu is supported alongside other targets
// CHECK:  warning: -fsycl-targets=native_cpu overrides SYCL targets option [-Wsycl-native-cpu-targets]
