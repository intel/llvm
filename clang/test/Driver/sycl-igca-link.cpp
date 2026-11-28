/// SYCL device code for Intel GPUs is compiled once per IGCA target. At link
/// time, SYCL device code for the same IGCA target and devices is linked
/// together, and the devices are passed on with --code.

// REQUIRES: x86-registered-target
// UNSUPPORTED: system-windows

// RUN: %clang --offload-new-driver -fsycl --target=x86_64-unknown-linux-gnu \
// RUN:   --offload-targets=spirv64-unknown-unknown --offload-arch=xe-mtl-u,xe-mtl-h -c %s -o %t-uh-a.o
// RUN: %clang --offload-new-driver -fsycl --target=x86_64-unknown-linux-gnu \
// RUN:   --offload-targets=spirv64-unknown-unknown --offload-arch=xe-mtl-h,xe-mtl-u -c %s -o %t-uh-b.o
// RUN: %clang --offload-new-driver -fsycl --target=x86_64-unknown-linux-gnu \
// RUN:   --offload-targets=spirv64-unknown-unknown --offload-arch=xe-mtl-u -c %s -o %t-u.o

/// xe-mtl-u and xe-mtl-h share igca_30r, so the SYCL device code of both
/// objects is linked together.
/// FIXME: sycl-web does not default -fsycl to the new offloading driver, and its
/// clang-linker-wrapper links SYCL device code in-process (llvm-link,
/// sycl-post-link, llvm-spirv) instead of calling clang --sycl-link with
/// -Xlinker --code=<device> as upstream #239010 does. The device list is
/// checked in the names of the extracted device images instead.
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t-uh-a.o %t-uh-b.o -o a.out 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SAME-DEVICES
// SAME-DEVICES:      spirv-to-ir-wrapper" {{[^ ]*}}.tmp-uh-a-spirv64-unknown-unknown-igca_30r.xe-mtl-h.xe-mtl-u-{{[^ ]*}}.bc -o [[UH_A:[^ ]*\.bc]]
// SAME-DEVICES-NEXT: spirv-to-ir-wrapper" {{[^ ]*}}.tmp-uh-b-spirv64-unknown-unknown-igca_30r.xe-mtl-h.xe-mtl-u-{{[^ ]*}}.bc -o [[UH_B:[^ ]*\.bc]]
// SAME-DEVICES-NEXT: llvm-link" --suppress-warnings [[UH_A]] [[UH_B]] -o
// SAME-DEVICES-NEXT: sycl-post-link"
// SAME-DEVICES-NEXT: llvm-spirv"
// SAME-DEVICES-NEXT: offload-wrapper:
// SAME-DEVICES-NOT:  spirv-to-ir-wrapper"
// SAME-DEVICES-NOT:  llvm-link"

/// An object compiled for xe-mtl-u only has a different list of devices, so its
/// SYCL device code is linked separately.
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t-u.o %t-uh-a.o -o a.out 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DIFFERENT-DEVICES
// DIFFERENT-DEVICES:      spirv-to-ir-wrapper" {{[^ ]*}}.tmp-u-spirv64-unknown-unknown-igca_30r.xe-mtl-u-{{[^ ]*}}.bc -o [[U:[^ ]*\.bc]]
// DIFFERENT-DEVICES-NEXT: llvm-link" --suppress-warnings [[U]] -o
// DIFFERENT-DEVICES-NEXT: sycl-post-link"
// DIFFERENT-DEVICES-NEXT: llvm-spirv"
// DIFFERENT-DEVICES-NEXT: offload-wrapper:
// DIFFERENT-DEVICES:      spirv-to-ir-wrapper" {{[^ ]*}}.tmp-uh-a-spirv64-unknown-unknown-igca_30r.xe-mtl-h.xe-mtl-u-{{[^ ]*}}.bc -o [[UH:[^ ]*\.bc]]
// DIFFERENT-DEVICES-NEXT: llvm-link" --suppress-warnings [[UH]] -o
// DIFFERENT-DEVICES-NEXT: sycl-post-link"
// DIFFERENT-DEVICES-NEXT: llvm-spirv"
// DIFFERENT-DEVICES-NEXT: offload-wrapper:
