// This test prepares Archive input for clang-offload-bundler
// and checks -exclude-target command line option.
// Option should exclude fat_device_aoco object file.

// UNSUPPORTED: system-windows

// The test uses assembled archive file fatlib.a.
// The assembly algorithm is the following:
// echo "DUMMY IR FILE" > device
// echo "DUMMY IR2 FILE" > device2
// echo "DUMMY AOCO FILE" > aoco
// echo "DUMMY HOST FILE" > host
// echo "DUMMY HOST2 FILE" > host2
// # Wrap and compile objects
// clang-offload-wrapper -o=device.bc -host=x86_64-unknown-linux-gnu -target=spir64 -kind=sycl device
// clang-offload-wrapper -o=device2.bc -host=x86_64-unknown-linux-gnu -target=spir64 -kind=sycl device2
// clang-offload-wrapper -o=aoco.bc -host=x86_64-unknown-linux-gnu -target=spir64 -kind=sycl aoco
// clang-offload-wrapper -o=host.bc -host=x86_64-unknown-linux-gnu -target=spir64 -kind=sycl host
// clang-offload-wrapper -o=host2.bc -host=x86_64-unknown-linux-gnu -target=spir64 -kind=sycl host2
// llc -filetype=obj -o device.o device.bc
// llc -filetype=obj -o device2.o device2.bc
// llc -filetype=obj -o aoco.o aoco.bc
// llc -filetype=obj -o host.o host.bc
// llc -filetype=obj -o host2.o host2.bc
// # Bundle the objects
// clang-offload-bundler -input=device.o -input=host.o -output=fat_device.o -targets=sycl-spir64_fpga-unknown-unknown,host-x86_64-unknown-linux-gnu -type=o
// clang-offload-bundler -input=device2.o -input=aoco.o -input=host2.o -output=fat_device_aoco.o -targets=sycl-spir64_fpga-unknown-unknown,sycl-fpga_aoco-intel-unknown,host-x86_64-unknown-linux-gnu -type=o
// # Create the archive
// ar cr fatlib.a fat_device.o fat_device_aoco.o


// Unbundle archive
// RUN: clang-offload-bundler -type=aoo -excluded-targets=sycl-fpga_aoco-intel-unknown -targets=sycl-spir64_fpga-unknown-unknown -input=%S/Inputs/clang-offload-bundler-exclude/fatlib.a -output=%t-my_output.txt -unbundle -allow-missing-bundles

// Check that output of unbundling doesn't contain content of device2
// RUN: cat %t-my_output.txt | xargs cat | strings | not grep "DUMMY IR2"
