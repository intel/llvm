// This test checks for the presence of target specific macros for openCL
//
// RUN: %clang_cc1 %s -E -dM -triple amdgcn-amdhsa-amdhsa \
// RUN: | FileCheck --check-prefix=CHECK-AMDGPU %s
// CHECK-AMDGPU: #define __HIP_MEMORY_SCOPE_AGENT
// CHECK-AMDGPU: #define __HIP_MEMORY_SCOPE_SINGLETHREAD
// CHECK-AMDGPU: #define __HIP_MEMORY_SCOPE_SYSTEM
// CHECK-AMDGPU: #define __HIP_MEMORY_SCOPE_WAVEFRONT
// CHECK-AMDGPU: #define __HIP_MEMORY_SCOPE_WORKGROUP
