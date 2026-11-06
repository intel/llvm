///
/// Test -fsycl-link with fat static archive inputs using the old offloading
/// model.  The device-only wrapped object must be the only output; the
/// archives must not cause an additional host link output.
///
// REQUIRES: x86-registered-target

// RUN: echo "void foo(void) {}" > %t1.cpp
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver %t1.cpp -c -o %t1_bundle.o

/// Archives along with an object: no 'multiple output files' error.
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver \
// RUN:   -fsycl-link -fPIC %t1_bundle.o %S/Inputs/SYCL/liblin64.a \
// RUN:   -o %t_prelinked.o -ccc-print-phases 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL-LINK-ARCHIVE
/// Archives only.
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver \
// RUN:   -fsycl-link -fPIC %S/Inputs/SYCL/liblin64.a \
// RUN:   -o %t_prelinked.o -ccc-print-phases 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL-LINK-ARCHIVE
// SYCL-LINK-ARCHIVE-NOT: cannot specify -o when generating multiple output files
// SYCL-LINK-ARCHIVE: clang-offload-unbundler, {{.*}}, tempfilelist
// SYCL-LINK-ARCHIVE: clang-offload-wrapper, {{.*}}, ir, (host-sycl)
// SYCL-LINK-ARCHIVE: {{[0-9]+}}: assembler, {{.*}}, object, (host-sycl)
// SYCL-LINK-ARCHIVE-NOT: linker, {{.*}}, (host-sycl)

/// Same check through the -### job list: the final job is the host compile of
/// the wrapped device image into the requested output, and no further job
/// (e.g. a host link of the archive) follows it.
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver \
// RUN:   -fsycl-link -fPIC %t1_bundle.o %S/Inputs/SYCL/liblin64.a \
// RUN:   -o %t_prelinked.o -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL-LINK-ARCHIVE-JOBS
// SYCL-LINK-ARCHIVE-JOBS-NOT: cannot specify -o when generating multiple output files
// SYCL-LINK-ARCHIVE-JOBS: clang-offload-wrapper{{.*}} "-o" "[[WRAPPED:[^"]+\.bc]]"
// SYCL-LINK-ARCHIVE-JOBS: clang{{.*}} "-cc1"{{.*}} "-o" "{{[^"]*}}_prelinked.o" {{.*}}"[[WRAPPED]]"
// SYCL-LINK-ARCHIVE-JOBS-NOT: {{^ "}}
