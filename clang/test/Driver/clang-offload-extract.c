// REQUIRES: x86-registered-target

//
// Check help message.
//
// RUN: clang-offload-extract --help | FileCheck %s --check-prefix CHECK-HELP

// CHECK-HELP: OVERVIEW: A tool for extracting target images from the linked fat offload binary.
// CHECK-HELP: USAGE: clang-offload-extract [options] <input file>
// CHECK-HELP: OPTIONS:
// CHECK-HELP: Generic Options:
// CHECK-HELP:   --help            - Display available options (--help-hidden for more)
// CHECK-HELP:   --help-list       - Display list of available options (--help-list-hidden for more)
// CHECK-HELP:   --version         - Display the version of this program
// CHECK-HELP: clang-offload-extract options:
// CHECK-HELP:   --output=<string> - Specifies prefix for the output file(s). Output file name
// CHECK-HELP:                       is composed from this prefix and the sequential number
// CHECK-HELP:                       of extracted image appended to the prefix.

//
// Create fat offload binary with two embedded target images.
//
// RUN: echo 'Target image 1' > %t.bin0
// RUN: echo 'Target image 2' > %t.bin1
// RUN: clang-offload-wrapper -kind=openmp -target=tg0 %t.bin0 -kind=sycl -target=tg1 %t.bin1 -o %t.wrapped.bc
// RUN: %clang %s %t.wrapped.bc -o %t.fat.bin

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
