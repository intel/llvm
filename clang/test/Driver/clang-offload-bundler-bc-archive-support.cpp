////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Make bundled object with targets:
// sycl-spir64-unknown-unknown
// host-x86_64-unknown-linux-gnu
// RUN: %clangxx -fsycl -c %s -o %t_bundled.o

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Make three distinct BC files
// RUN: %clangxx -fsycl -fsycl-device-only -DTYPE1 %s -o %t1.bc
// RUN: %clangxx -fsycl -fsycl-device-only -DTYPE2 %s -o %t2.bc
// RUN: %clangxx -fsycl -fsycl-device-only -DTYPE3 %s -o %t3.bc

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Bundle BC files to different targets:
// host-spir64-unknown-unknown
// host-spir64_gen
// host-spir64_x86_64
// RUN: clang-offload-bundler -type=bc -targets=host-spir64-unknown-unknown -input=%t1.bc  -output=%t1_bundled.bc
// RUN: clang-offload-bundler -type=bc -targets=host-spir64_gen             -input=%t2.bc  -output=%t2_bundled.bc
// RUN: clang-offload-bundler -type=bc -targets=host-spir64_x86_64          -input=%t3.bc  -output=%t3_bundled.bc

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Make archive with bundled BC and o files
// RUN: rm -f %t_bundled.a
// RUN: ar cr %t_bundled.a %t1_bundled.bc %t2_bundled.bc %t3_bundled.bc %t_bundled.o

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Check that -list with various archive types can find all targets
// RUN: clang-offload-bundler -list -type=ao  -input=%t_bundled.a | FileCheck --check-prefixes=CHECK-LIST %s
// RUN: clang-offload-bundler -list -type=aoo -input=%t_bundled.a | FileCheck --check-prefixes=CHECK-LIST %s

// CHECK-LIST-DAG: sycl-spir64-unknown-unknown
// CHECK-LIST-DAG: host-x86_64-unknown-linux-gnu
// CHECK-LIST-DAG: host-spir64-unknown-unknown
// CHECK-LIST-DAG: host-spir64_gen
// CHECK-LIST-DAG: host-spir64_x86_64

// RUN: clang-offload-bundler -list -type=ao  -input=%t_bundled.a | wc -l | FileCheck --check-prefixes=CHECK-LIST-LENGTH %s
// RUN: clang-offload-bundler -list -type=aoo -input=%t_bundled.a | wc -l | FileCheck --check-prefixes=CHECK-LIST-LENGTH %s

// CHECK-LIST-LENGTH: 5

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Test -check-section
// RUN:     clang-offload-bundler -check-section -type=ao  -input=%t_bundled.a -targets=sycl-spir64-unknown-unknown
// RUN:     clang-offload-bundler -check-section -type=ao  -input=%t_bundled.a -targets=host-x86_64-unknown-linux-gnu
// RUN:     clang-offload-bundler -check-section -type=ao  -input=%t_bundled.a -targets=host-spir64-unknown-unknown
// RUN:     clang-offload-bundler -check-section -type=ao  -input=%t_bundled.a -targets=host-spir64_gen
// RUN:     clang-offload-bundler -check-section -type=ao  -input=%t_bundled.a -targets=host-spir64_x86_64
// RUN: not clang-offload-bundler -check-section -type=ao  -input=%t_bundled.a -targets=sycl-spir64-unknown-unknown-a
// RUN: not clang-offload-bundler -check-section -type=ao  -input=%t_bundled.a -targets=host-x86_64-unknown-linux-gnu-b
// RUN: not clang-offload-bundler -check-section -type=ao  -input=%t_bundled.a -targets=host-spir64-unknown-unknown-c
// RUN: not clang-offload-bundler -check-section -type=ao  -input=%t_bundled.a -targets=host-spir64_gen-d
// RUN: not clang-offload-bundler -check-section -type=ao  -input=%t_bundled.a -targets=host-spir64_x86_64-e
// RUN:     clang-offload-bundler -check-section -type=aoo -input=%t_bundled.a -targets=sycl-spir64-unknown-unknown
// RUN:     clang-offload-bundler -check-section -type=aoo -input=%t_bundled.a -targets=host-x86_64-unknown-linux-gnu
// RUN:     clang-offload-bundler -check-section -type=aoo -input=%t_bundled.a -targets=host-spir64-unknown-unknown
// RUN:     clang-offload-bundler -check-section -type=aoo -input=%t_bundled.a -targets=host-spir64_gen
// RUN:     clang-offload-bundler -check-section -type=aoo -input=%t_bundled.a -targets=host-spir64_x86_64
// RUN: not clang-offload-bundler -check-section -type=aoo -input=%t_bundled.a -targets=sycl-spir64-unknown-unknown-a
// RUN: not clang-offload-bundler -check-section -type=aoo -input=%t_bundled.a -targets=host-x86_64-unknown-linux-gnu-b
// RUN: not clang-offload-bundler -check-section -type=aoo -input=%t_bundled.a -targets=host-spir64-unknown-unknown-c
// RUN: not clang-offload-bundler -check-section -type=aoo -input=%t_bundled.a -targets=host-spir64_gen-d
// RUN: not clang-offload-bundler -check-section -type=aoo -input=%t_bundled.a -targets=host-spir64_x86_64-e

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Unbundle object file to use as a reference result
// RUN: clang-offload-bundler -unbundle -type=o -input=%t_bundled.o -targets=sycl-spir64-unknown-unknown   -output=%t_unbundled_A.o
// RUN: clang-offload-bundler -unbundle -type=o -input=%t_bundled.o -targets=host-x86_64-unknown-linux-gnu -output=%t_unbundled_B.o

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Test archive unbundling
// RUN: clang-offload-bundler -unbundle -type=aoo -input=%t_bundled.a -targets=sycl-spir64-unknown-unknown   -output=%t_list1.txt
// RUN: clang-offload-bundler -unbundle -type=aoo -input=%t_bundled.a -targets=host-x86_64-unknown-linux-gnu -output=%t_list2.txt
// RUN: clang-offload-bundler -unbundle -type=aoo -input=%t_bundled.a -targets=host-spir64-unknown-unknown   -output=%t_list3.txt
// RUN: clang-offload-bundler -unbundle -type=aoo -input=%t_bundled.a -targets=host-spir64_gen               -output=%t_list4.txt
// RUN: clang-offload-bundler -unbundle -type=aoo -input=%t_bundled.a -targets=host-spir64_x86_64            -output=%t_list5.txt

// RUN: cmp %t_unbundled_A.o `cat %t_list1.txt`
// RUN: cmp %t_unbundled_B.o `cat %t_list2.txt`
// RUN: cmp %t1.bc           `cat %t_list3.txt`
// RUN: cmp %t2.bc           `cat %t_list4.txt`
// RUN: cmp %t3.bc           `cat %t_list5.txt`

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Test archive unbundling for multiple targets
// RUN: clang-offload-bundler -unbundle -type=aoo -input=%t_bundled.a -targets=sycl-spir64-unknown-unknown,host-spir64_gen -output=%t_listA.txt -output=%t_listB.txt

// RUN: cmp %t_unbundled_A.o `cat %t_listA.txt`
// RUN: cmp %t2.bc           `cat %t_listB.txt`

#include <sycl/sycl.hpp>
#include <iostream>

using namespace sycl;

const std::string secret { "Ifmmp-!xpsme\"\012J"};

const auto sz = secret.size();

int main() {
  queue Q;

  char *result = malloc_shared<char>(sz,Q);

  std::memcpy(result,secret.data(),sz);

  Q.parallel_for(sz,[=](auto&i) {
#ifdef TYPE1
    result[i] -= 13;
#elif TYPE2
    result[i] *= 17;
#elif TYPE3
    result[i] += 23;
#endif
    
  }).wait();

  std::cout << result << "\n";
  return 0;
}
