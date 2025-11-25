// Ensure that the profile update mode is set to 'atomic' when compiling SYCL code
// RUN: %clang -### -fsycl -fprofile-instr-generate -fcoverage-mapping %s 2>&1 | FileCheck %s
// CHECK: "-fprofile-update=atomic"
