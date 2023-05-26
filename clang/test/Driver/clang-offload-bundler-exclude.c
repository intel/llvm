// This test prepares Archive input for clang-offload-bundler
// and checks -exclude-target command line option.
// Option should exclude fat_device_aoco object file.

// UNSUPPORTED: system-windows

// RUN: echo "DUMMY IR FILE" > %t-device
// RUN: echo "DUMMY IR2 FILE" > %t-device2
// RUN: echo "DUMMY AOCO FILE" > %t-aoco
// RUN: echo "DUMMY HOST FILE" > %t-host
// RUN: echo "DUMMY HOST2 FILE" > %t-host2

// Wrap and compile objects
// RUN: clang-offload-wrapper -o=%t-device.bc -host=x86_64-unknown-linux-gnu -target=spir64 -kind=sycl %t-device
// RUN: clang-offload-wrapper -o=%t-device2.bc -host=x86_64-unknown-linux-gnu -target=spir64 -kind=sycl %t-device2
// RUN: clang-offload-wrapper -o=%t-aoco.bc -host=x86_64-unknown-linux-gnu -target=spir64 -kind=sycl %t-aoco
// RUN: clang-offload-wrapper -o=%t-host.bc -host=x86_64-unknown-linux-gnu -target=spir64 -kind=sycl %t-host
// RUN: clang-offload-wrapper -o=%t-host2.bc -host=x86_64-unknown-linux-gnu -target=spir64 -kind=sycl %t-host2

// RUN: llc -filetype=obj -o %t-device.o %t-device.bc
// RUN: llc -filetype=obj -o %t-device2.o %t-device2.bc
// RUN: llc -filetype=obj -o %t-aoco.o %t-aoco.bc
// RUN: llc -filetype=obj -o %t-host.o %t-host.bc
// RUN: llc -filetype=obj -o %t-host2.o %t-host2.bc

// Bundle the objects
// RUN: clang-offload-bundler -input=%t-device.o -input=%t-host.o -output=%t-fat_device.o -targets=sycl-spir64_fpga-unknown-unknown,host-x86_64-unknown-linux-gnu -type=o

// RUN: clang-offload-bundler -input=%t-device2.o -input=%t-aoco.o -input=%t-host2.o -output=%t-fat_device_aoco.o -targets=sycl-spir64_fpga-unknown-unknown,sycl-fpga_aoco-intel-unknown,host-x86_64-unknown-linux-gnu -type=o

// Create the archive
// RUN: ar cr %t-fatlib.a %t-fat_device.o %t-fat_device_aoco.o

// RUN: clang-offload-bundler -type=aoo -excluded-targets=sycl-fpga_aoco-intel-unknown -targets=sycl-spir64_fpga-unknown-unknown -input=%t-fatlib.a -output=%t-my_output.txt -unbundle -allow-missing-bundles

// Check that output of unbundling doesn't contain content of device2
// RUN: cat %t-my_output.txt | xargs cat | strings | not grep "DUMMY IR2"
