// This test checks that the packaging (create) and extraction (use) sides of
// a Windows offload-binary agree on the exact host triple, including the
// MSVC toolset version suffix (e.g. msvc19.29.30159) computed by
// ComputeEffectiveClangTriple(). A mismatch here means llvm-offload-binary's
// exact-triple match fails silently, producing an empty extracted file (as
// happened with PCH extraction).
//
// The extraction path is exercised here by feeding a packaged offload-binary
// preprocessed (.ii) file back into the driver, mirroring how PCH extraction
// invokes OffloadPackagerExtract::ConstructJob.

// RUN: %clang --target=x86_64-pc-windows-msvc -fms-compatibility-version=19.29.30159 \
// RUN:   --offload-new-driver -fsycl -E -o %t.ii %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix CREATE %s
// CREATE: llvm-offload-binary{{.*}} "-o" "{{.*}}.ii"
// CREATE-SAME: "--image=file={{.*}}.ii,triple=spir64-unknown-unknown,arch=generic,kind=sycl"
// CREATE-SAME: "--image=file={{.*}}.ii,triple=x86_64-pc-windows-msvc19.29.30159,arch=generic,kind=host"

// RUN: %clang --target=x86_64-pc-windows-msvc -fms-compatibility-version=19.29.30159 \
// RUN:   --offload-new-driver -fsycl -E -o %t.ii %s
// RUN: %clang --target=x86_64-pc-windows-msvc -fms-compatibility-version=19.29.30159 \
// RUN:   --offload-new-driver -fsycl -c %t.ii -### 2>&1 \
// RUN:   | FileCheck -check-prefix USE %s
// USE: llvm-offload-binary{{.*}} "[[MAINFILE:.+\.ii]]"
// USE-SAME: "--image=file={{.*}}.ii,triple=spir64-unknown-unknown,arch=generic,kind=sycl"
// USE: llvm-offload-binary{{.*}} "[[MAINFILE]]"
// USE-SAME: "--image=file={{.*}}.ii,triple=x86_64-pc-windows-msvc19.29.30159,arch=generic,kind=host"

int main() { return 0; }
