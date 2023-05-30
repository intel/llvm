// REQUIRES: x86-registered-target

//
// Check help message.
//
// RUN: clang-offload-extract --help | FileCheck %s --check-prefix CHECK-HELP

// CHECK-HELP: OVERVIEW:
// CHECK-HELP: 
// CHECK-HELP: A utility to extract all the target images from a
// CHECK-HELP: linked fat binary, and store them in separate files.
// CHECK-HELP: 
// CHECK-HELP: USAGE: clang-offload-extract [options] <input file>
// CHECK-HELP: 
// CHECK-HELP: OPTIONS:
// CHECK-HELP: 
// CHECK-HELP: Generic Options:
// CHECK-HELP: 
// CHECK-HELP:   --help          - Display available options (--help-hidden for more)
// CHECK-HELP:   --help-list     - Display list of available options (--help-list-hidden for more)
// CHECK-HELP:   --version       - Display the version of this program
// CHECK-HELP: 
// CHECK-HELP: Utility-specific options:
// CHECK-HELP: 
// CHECK-HELP:   --stem=<string> - Specifies the stem for the output file(s).
// CHECK-HELP:                     The default stem when not specified is "target.bin".
// CHECK-HELP:                     The Output file name is composed from this stem and
// CHECK-HELP:                     the sequential number of each extracted image appended
// CHECK-HELP:                     to the stem:
// CHECK-HELP:                       <stem>.<index>

//
// Create fat offload binary with two embedded target images.
//
// RUN: echo 'Target image 1' > %t.bin0
// RUN: echo 'Target image 2' > %t.bin1
// RUN: clang-offload-wrapper -kind=openmp -target=tg0 %t.bin0 -kind=sycl -target=tg1 %t.bin1 -o %t.wrapped.bc
// RUN: %clang -fdeclspec %s %t.wrapped.bc -o %t.fat.bin

//
// Extract target images.
//
// RUN: clang-offload-extract --output=%t.extracted %t.fat.bin | FileCheck %s --check-prefix CHECK-EXTRACT
// CHECK-EXTRACT: Saving target image to
// CHECK-EXTRACT: Saving target image to

//
// Check that extracted contents match the original images.
//
// RUN: diff %t.extracted.0 %t.bin0
// RUN: diff %t.extracted.1 %t.bin1

//
// Some code so that we can build an offload executable from this file.
//
#pragma section(".tgtimg", read)
__declspec(allocate(".tgtimg"))
__declspec(align(sizeof(void*) * 2))
const void* padding[2] = {0, 0};

#ifdef _WIN32
char __start_omp_offloading_entries = 1;
char __stop_omp_offloading_entries = 1;
#endif

void __tgt_register_lib(void *desc) {}
void __tgt_unregister_lib(void *desc) {}

void __sycl_register_lib(void* desc) {}
void __sycl_unregister_lib(void* desc) {}

int main(void) {
  return 0;
}
