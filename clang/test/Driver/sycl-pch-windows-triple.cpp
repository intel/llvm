// This test checks that the packaging (create) and extraction (use) sides of
// a Windows PCH agree on the exact host triple, including the MSVC toolset
// version suffix (e.g. msvc19.29.30159) computed by
// ComputeEffectiveClangTriple(). A mismatch here means llvm-offload-binary's
// exact-triple match fails silently, producing an empty extracted PCH.

// RUN: echo "// Header file" > %t.h
// RUN: %clang_cl --target=x86_64-pc-windows-msvc -fms-compatibility-version=19.29.30159 \
// RUN:   --offload-new-driver -fsycl -x c++-header %t.h -### 2>&1 \
// RUN:   | FileCheck -check-prefix=CREATE %s
// CREATE: llvm-offload-binary{{.*}} "-o" "{{.*}}.pch"
// CREATE-SAME: "--image=file={{.*}}.pch,triple=spir64-unknown-unknown,arch=generic,kind=sycl"
// CREATE-SAME: "--image=file={{.*}}.pch,triple=x86_64-pc-windows-msvc19.29.30159,arch=generic,kind=host"

// RUN: touch %t.pch
// RUN: %clang --target=x86_64-pc-windows-msvc -fms-compatibility-version=19.29.30159 \
// RUN:   --offload-new-driver -fsycl -c -include-pch %t.pch -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=USE %s
// USE: llvm-offload-binary{{.*}} "[[MAINPCHFILE:.+\.pch]]"
// USE-SAME: "--image=file={{.*}}.pch,triple=x86_64-pc-windows-msvc19.29.30159,arch=generic,kind=host"
// USE: llvm-offload-binary{{.*}} "[[MAINPCHFILE]]"
// USE-SAME: "--image=file={{.*}}.pch,triple=spir64-unknown-unknown,arch=generic,kind=sycl"
