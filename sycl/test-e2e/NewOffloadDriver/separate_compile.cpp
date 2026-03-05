//==--------------- separate_compile.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This test is copied from SeparateCompile/test.cpp
// and modified to test with the New Offloading Model.

// REQUIRES: target-spir
// >> ---- compile src1
// >> device compilation...
// RUN: %{run-aux} %clangxx --offload-new-driver -DSYCL_DISABLE_FALLBACK_ASSERT -fno-sycl-dead-args-optimization -fsycl-device-only -Xclang -fsycl-int-header=sycl_ihdr_a.h %s -o a_kernel.bc -Wno-sycl-strict
// >> host compilation...
// RUN: %{run-aux} %clangxx --offload-new-driver -Wno-error=ignored-attributes %sycl_include -Wno-error=unused-command-line-argument -DSYCL_DISABLE_FALLBACK_ASSERT %cxx_std_optionc++17 %include_option sycl_ihdr_a.h %debug_option -c %s -o a.o %sycl_options -fno-sycl-dead-args-optimization -Wno-sycl-strict
//
// >> ---- compile src2
// >> device compilation...
// RUN: %{run-aux} %clangxx --offload-new-driver -Wno-error=unused-command-line-argument -DSYCL_DISABLE_FALLBACK_ASSERT -DB_CPP=1 -fno-sycl-dead-args-optimization -fsycl-device-only -Xclang -fsycl-int-header=sycl_ihdr_b.h %s -o b_kernel.bc -Wno-sycl-strict
// >> host compilation...
// RUN:%{run-aux} %clangxx  --offload-new-driver -Wno-error=ignored-attributes %sycl_include -Wno-error=unused-command-line-argument -DSYCL_DISABLE_FALLBACK_ASSERT -DB_CPP=1 %cxx_std_optionc++17 %include_option sycl_ihdr_b.h %debug_option -c %s -o b.o %sycl_options -fno-sycl-dead-args-optimization -Wno-sycl-strict
//
// >> ---- link device code
// RUN: llvm-link -o=app.bc a_kernel.bc b_kernel.bc %sycl_static_libs_dir/libsycl-itt-compiler-wrappers.bc %sycl_static_libs_dir/libsycl-itt-stubs.bc %sycl_static_libs_dir/libsycl-itt-user-wrappers.bc


// >> ---- produce entries data
// RUN: sycl-post-link -split=auto -emit-param-info -symbols -emit-exported-symbols -o test.table app.bc
//
// >> ---- do table transformations from bc to spv entries
// RUN: file-table-tform -extract=Code -drop_titles -o test_spv_in.table test.table
// RUN: llvm-foreach --in-file-list=test_spv_in.table --in-replace=test_spv_in.table --out-ext=spv --out-file-list=test_spv_out.table --out-replace=test_spv_out.table -- llvm-spirv -o test_spv_out.table -spirv-allow-extra-diexpressions -spirv-allow-unknown-intrinsics=llvm.genx. -spirv-ext=-all test_spv_in.table
// RUN: file-table-tform -replace=Code,Code -o test_spv.table test.table test_spv_out.table
//
// >> ---- wrap device binary
// >> produce .bc
// RUN: clang-offload-wrapper -o wrapper.bc -host=x86_64 -kind=sycl -target=spir64 -batch test_spv.table
//
// >> compile .bc to .o
// RUN: %{run-aux} %clangxx --offload-new-driver -Wno-error=override-module -c wrapper.bc -o wrapper.o %if preview-mode %{-Wno-unused-command-line-argument%}
//
// >> ---- link the full hetero app
// RUN:%{run-aux} %clangxx wrapper.o a.o b.o -Wno-unused-command-line-argument -o app.exe %sycl_options
// RUN: %{run} ./app.exe
