///
/// Tests for using SPIR-V backend for SYCL offloading
///
// RUN: %clangxx -fsycl -fsycl-use-spirv-backend-for-spirv-gen -### %s 2>&1 | FileCheck %s

// CHECK: llc{{.*}} "-filetype=obj" "-mtriple=spirv64-unknown-unknown" "-O0" "--avoid-spirv-capabilities=Shader" "--translator-compatibility-mode" "--spirv-ext=
