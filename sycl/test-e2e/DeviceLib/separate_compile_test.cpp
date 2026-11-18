//===----------------------separate_compile_test.cpp-----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Tests separate compilation and linking of SYCL host and device code using
/// single- and double-precision complex math workloads. Verifies that the
/// resulting executables run correctly.
///
//===----------------------------------------------------------------------===//
// TODO: Remove 'UNSUPPORTED' below once
// https://github.com/llvm/llvm-project/pull/218528 lands and cherry-picked to
// intel/llvm.
// UNSUPPORTED: new-offload-model
// UNSUPPORTED-TRACKER: CMPLRLLVM-78263
// UNSUPPORTED: target-amd
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/22300

// XFAIL: target-native_cpu
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/20142

// DEFINE: %{mathflags} = %if cl_options %{/clang:-fno-fast-math%} %else %{-fno-fast-math%}

// >> device build and intergation header generation...
// RUN: %{build} -fsycl-link %S/std_complex_math_test.cpp -o %t_device.o %{mathflags}
// RUN: %clangxx %{sycl_target_opts} -fsycl-device-only        \
// RUN: -Xclang -fsycl-int-header=std_complex_math_test_ihdr.h \
// RUN: %S/std_complex_math_test.cpp -Wno-sycl-strict %{mathflags}
// >> host compilation...
// RUN: %clangxx %cxx_std_optionc++17 -Wno-error=unused-command-line-argument %sycl_include -Wno-error=ignored-attributes \
// RUN: %include_option std_complex_math_test_ihdr.h -c %S/std_complex_math_test.cpp                 \
// RUN: -o %t_host.o %sycl_options -Wno-sycl-strict %{mathflags}
// >> linking...
// RUN: %clangxx %t_host.o %t_device.o -Wno-unused-command-line-argument -o %t.out %sycl_options %{mathflags}
// RUN: %{run} %t.out

// >> device build and intergation header generation...
// RUN: %{build} -fsycl-link  %S/std_complex_math_fp64_test.cpp -o %t_fp64_device.o %{mathflags}
// RUN: %clangxx %{sycl_target_opts} -fsycl-device-only             \
// RUN: -Xclang -fsycl-int-header=std_complex_math_fp64_test_ihdr.h \
// RUN: %S/std_complex_math_fp64_test.cpp -Wno-sycl-strict %{mathflags}
// >> host compilation...
// RUN: %clangxx %cxx_std_optionc++17 -Wno-error=unused-command-line-argument %sycl_include -Wno-error=ignored-attributes \
// RUN: %include_option std_complex_math_fp64_test_ihdr.h -c %S/std_complex_math_fp64_test.cpp       \
// RUN: -o %t_fp64_host.o %sycl_options -Wno-sycl-strict %{mathflags}
// >> linking...
// RUN: %clangxx %t_fp64_host.o %t_fp64_device.o -Wno-unused-command-line-argument -o %t_fp64.out %sycl_options %{mathflags}
// RUN: %{run} %t_fp64.out
