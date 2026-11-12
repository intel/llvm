// This test prepares Archive input for clang-offload-bundler
// and checks -excluded-targets command line option.
// Option should exclude every archive member that contains an excluded target
// (fat_device_excl.o below), even if it also contains the requested target.

// UNSUPPORTED: system-windows
// REQUIRES: x86-registered-target

// Build the archive: two fat objects, the second one also has the excluded
// sycl-spir64_gen target.
// RUN: rm -rf %t && mkdir %t
// RUN: echo "DUMMY IR FILE" > %t/device
// RUN: echo "DUMMY IR2 FILE" > %t/device2
// RUN: echo "DUMMY GEN FILE" > %t/gen
// RUN: %clang -target x86_64-unknown-linux-gnu -c %s -o %t/host.o
// RUN: clang-offload-bundler -type=o -targets=sycl-spir64_x86_64-unknown-unknown,host-x86_64-unknown-linux-gnu -input=%t/device -input=%t/host.o -output=%t/fat_device.o
// RUN: clang-offload-bundler -type=o -targets=sycl-spir64_x86_64-unknown-unknown,sycl-spir64_gen-unknown-unknown,host-x86_64-unknown-linux-gnu -input=%t/device2 -input=%t/gen -input=%t/host.o -output=%t/fat_device_excl.o
// RUN: llvm-ar cr %t/fatlib.a %t/fat_device.o %t/fat_device_excl.o

// Unbundle archive
// RUN: clang-offload-bundler -type=aoo -excluded-targets=sycl-spir64_gen-unknown-unknown -targets=sycl-spir64_x86_64-unknown-unknown -input=%t/fatlib.a -output=%t/my_output.txt -unbundle -allow-missing-bundles

// Check that output of unbundling contains content of device, but not device2
// RUN: cat %t/my_output.txt | xargs cat | strings | grep "DUMMY IR FILE"
// RUN: cat %t/my_output.txt | xargs cat | strings | not grep "DUMMY IR2"

// Without -excluded-targets, device2 is extracted too
// RUN: clang-offload-bundler -type=aoo -targets=sycl-spir64_x86_64-unknown-unknown -input=%t/fatlib.a -output=%t/all_output.txt -unbundle -allow-missing-bundles
// RUN: cat %t/all_output.txt | xargs cat | strings | grep "DUMMY IR2"

void foo(void) {}
