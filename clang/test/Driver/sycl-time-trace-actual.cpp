// Test that -ftime-trace actually generates trace files for SYCL compilation.
// This test performs actual compilation
// and verifies that both host and device trace JSON files are created.

// REQUIRES: system-linux

// Setup: Create test directories and input file
// RUN: rm -rf %t && mkdir -p %t/src %t/traces
// RUN: cp %s %t/src/test.cpp

// Test 1: Compile-only mode with explicit trace directory
// RUN: %clang --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:   -c -ftime-trace=%t/traces -ftime-trace-granularity=0 \
// RUN:   -ftime-trace-verbose %t/src/test.cpp -o %t/test.o

// Verify host and device trace files were generated
// RUN: ls %t/traces/ | FileCheck %s --check-prefix=CHECK-TEST1
// CHECK-TEST1-DAG: test-host-x86_64-unknown-linux-gnu.json
// CHECK-TEST1-DAG: test-sycl-spir64-unknown-unknown.json

// Validate that trace files contain valid JSON with traceEvents
// RUN: cat %t/traces/test-host-x86_64-unknown-linux-gnu.json | FileCheck %s --check-prefix=CHECK-JSON
// RUN: cat %t/traces/test-sycl-spir64-unknown-unknown.json | FileCheck %s --check-prefix=CHECK-JSON
// CHECK-JSON: "traceEvents"

// Test 2: Verify trace file naming with -o option
// When using -c with -o, trace files should be named based on the -o output
// RUN: rm -rf %t/traces2 && mkdir -p %t/traces2
// RUN: %clang --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:   -c -ftime-trace=%t/traces2 %t/src/test.cpp -o %t/different_name.o

// Should be named after the -o output (different_name), not the source (test.cpp)
// RUN: ls %t/traces2/ | FileCheck %s --check-prefix=CHECK-TEST2
// CHECK-TEST2-DAG: different_name-host-x86_64-unknown-linux-gnu.json
// CHECK-TEST2-DAG: different_name-sycl-spir64-unknown-unknown.json

// Test 3: Compile and link mode
// In compile+link mode, trace files are named based on source file
// RUN: rm -rf %t/traces3 && mkdir -p %t/traces3
// RUN: %clang --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:   -ftime-trace=%t/traces3 %t/src/test.cpp -o %t/myapp

// Frontend traces should still be generated (named after source file)
// RUN: ls %t/traces3/ | FileCheck %s --check-prefix=CHECK-TEST3
// CHECK-TEST3-DAG: test-host-x86_64-unknown-linux-gnu.json
// CHECK-TEST3-DAG: test-sycl-spir64-unknown-unknown.json

// Test 4: Multiple input files with unique basenames
// RUN: rm -rf %t/traces4 && mkdir -p %t/traces4 %t/src2
// RUN: cp %s %t/src2/file1.cpp
// RUN: cp %s %t/src2/file2.cpp
// RUN: %clang --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:   -c -ftime-trace=%t/traces4 %t/src2/file1.cpp %t/src2/file2.cpp

// Should have clean trace file names since basenames are unique
// RUN: ls %t/traces4/ | FileCheck %s --check-prefix=CHECK-UNIQUE
// CHECK-UNIQUE-DAG: file1-host-x86_64-unknown-linux-gnu.json
// CHECK-UNIQUE-DAG: file1-sycl-spir64-unknown-unknown.json
// CHECK-UNIQUE-DAG: file2-host-x86_64-unknown-linux-gnu.json
// CHECK-UNIQUE-DAG: file2-sycl-spir64-unknown-unknown.json

// Test 5: Multiple input files with basename collision (directory name used)
// When basenames collide, traces use parent directory name to disambiguate
// RUN: rm -rf %t/traces5 && mkdir -p %t/traces5 %t/dir1 %t/dir2
// RUN: cp %s %t/dir1/test.cpp
// RUN: cp %s %t/dir2/test.cpp
// RUN: %clang --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:   -c -ftime-trace=%t/traces5 %t/dir1/test.cpp %t/dir2/test.cpp

// Should have 4 trace files (2 sources × 2 traces each)
// Verify they use directory names to disambiguate (dir1-test, dir2-test)
// RUN: ls %t/traces5/ | FileCheck %s --check-prefix=CHECK-COLLISION
// CHECK-COLLISION-DAG: dir1-test-host-x86_64-unknown-linux-gnu.json
// CHECK-COLLISION-DAG: dir1-test-sycl-spir64-unknown-unknown.json
// CHECK-COLLISION-DAG: dir2-test-host-x86_64-unknown-linux-gnu.json
// CHECK-COLLISION-DAG: dir2-test-sycl-spir64-unknown-unknown.json

// Minimal SYCL code for testing
int main() {
  return 0;
}
