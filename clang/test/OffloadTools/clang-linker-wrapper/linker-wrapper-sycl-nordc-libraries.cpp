// REQUIRES: system-linux, x86-registered-target

// Each no-RDC user module must resolve device libraries without linking
// independent translation units together. Two old-model objects (compiled
// without --offload-new-driver, so device libs aren't linked in at compile
// time) are used so clang-linker-wrapper's old-model device-lib fallback
// (see getInputBitcodeLibrary) actually applies here.
// RUN: %clang -fsycl --no-offload-new-driver -fsycl-targets=spir64-unknown-unknown -c %s -o %t.o
// RUN: cp %t.o %t2.o
// RUN: touch %t.lib.bc
// RUN: clang-linker-wrapper --dry-run --emit-fatbin-only --no-sycl-rdc --bitcode-library=spir64-unknown-unknown=%t.lib.bc --linker-path=/usr/bin/ld %t.o %t2.o -o %t.fatbin 2>&1 | FileCheck %s --check-prefix=NORDC --implicit-check-not='llvm-link" --suppress-warnings'
// NORDC: llvm-link" --only-needed --suppress-warnings {{.*}} [[LIB:.*\.lib\.bc]] -o [[FIRST:.*\.bc]]
// NORDC: llvm-link" --only-needed --suppress-warnings {{.*}} [[LIB]] -o [[SECOND:.*\.bc]]
// NORDC-NOT: llvm-link
// NORDC: sycl-post-link{{.*}} [[FIRST]]
// NORDC: sycl-post-link{{.*}} [[SECOND]]

// RDC still combines user inputs before resolving device libraries once.
// RUN: clang-linker-wrapper --dry-run --emit-fatbin-only --bitcode-library=spir64-unknown-unknown=%t.lib.bc --linker-path=/usr/bin/ld %t.o %t2.o -o %t.rdc.fatbin 2>&1 | FileCheck %s --check-prefix=RDC
// RDC: llvm-link" --suppress-warnings {{.*}}.bc {{.*}}.bc -o [[LINKED:.*\.bc]]
// RDC-NEXT: {{.*}}llvm-link" --only-needed --suppress-warnings [[LINKED]] {{.*}}.lib.bc -o [[LIBLINKED:.*\.bc]]
// RDC: sycl-post-link{{.*}} [[LIBLINKED]]
// RDC-NOT: sycl-post-link

// Unmatched libraries are ignored; matched missing libraries report an error
// in no-RDC mode too.
// RUN: clang-linker-wrapper --dry-run --emit-fatbin-only --no-sycl-rdc --bitcode-library=nvptx64-nvidia-cuda=%t.missing.bc --linker-path=/usr/bin/ld %t.o %t2.o -o %t.unmatched.fatbin 2>&1 | FileCheck %s --check-prefix=UNMATCHED
// UNMATCHED-NOT: llvm-link
// UNMATCHED: sycl-post-link
// UNMATCHED-NOT: llvm-link
// RUN: not clang-linker-wrapper --dry-run --emit-fatbin-only --no-sycl-rdc --bitcode-library=spir64-unknown-unknown=%t.missing.bc --linker-path=/usr/bin/ld %t.o %t2.o -o %t.missing.fatbin 2>&1 | FileCheck %s --check-prefix=MISSING
// MISSING: error: The specified device library {{.*}}.missing.bc does not exist.
