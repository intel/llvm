///
/// Test linking with -fsycl an object created with -fsycl-link using the old
/// offloading model. Such an object holds already wrapped device images (in a
/// bundle-like section such as __CLANG_OFFLOAD_BUNDLE__sycl-spir64) but no host
/// bundle. It must be passed to the host linker as is: it must not be unbundled
/// and its device images must not be sent through the device link again.
///
// REQUIRES: x86-registered-target

/// Create a stand-in for an -fsycl-link output: a host object with a SYCL
/// device image section and no host bundle.
// RUN: echo "void foo(void) {}" > %t_host.cpp
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -c %t_host.cpp -o %t_host.o
// RUN: echo "dummy device image" > %t_image.spv
// RUN: llvm-objcopy --add-section=__CLANG_OFFLOAD_BUNDLE__sycl-spir64=%t_image.spv \
// RUN:   %t_host.o %t_prelinked.o
// RUN: clang-offload-bundler -type=o -input=%t_prelinked.o -list \
// RUN:   | FileCheck %s -check-prefix=SECTIONS --implicit-check-not=host-
// SECTIONS: sycl-spir64

/// Fat object from a regular -fsycl compilation.
// RUN: echo "void bar(void) {}" > %t_fat.cpp
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver \
// RUN:   -c %t_fat.cpp -o %t_fat.o

/// Only the fat object goes through the device link; the pre-linked object is
/// passed to the host linker unchanged, for executables and shared libraries.
/// File names are matched instead of full paths, as -### escapes backslashes
/// in Windows paths.
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver \
// RUN:   %t_prelinked.o %t_fat.o -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CHECK-LINK \
// RUN:       --implicit-check-not='"-input={{.*}}_prelinked.o" {{.*}}"-unbundle"'
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver \
// RUN:   -shared %t_prelinked.o %t_fat.o -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CHECK-LINK \
// RUN:       --implicit-check-not='"-input={{.*}}_prelinked.o" {{.*}}"-unbundle"'
// CHECK-LINK: clang-offload-bundler{{.*}} "-input={{.*}}_fat.o" {{.*}}"-unbundle"
// CHECK-LINK: clang-offload-wrapper
// CHECK-LINK: "{{.*}}ld{{(.exe)?}}" {{.*}}"{{.*}}_prelinked.o"

/// -fsycl-link still consumes the device images of a pre-linked input.
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver \
// RUN:   -fsycl-link %t_prelinked.o %t_fat.o -o %t_out.o -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CHECK-SYCL-LINK
// CHECK-SYCL-LINK: clang-offload-bundler{{.*}} "-input={{.*}}_prelinked.o" {{.*}}"-unbundle"
// CHECK-SYCL-LINK: clang-offload-wrapper
