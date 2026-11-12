// Verify that a compilation unit ID (CUID) is generated for SYCL compilations
// and that the same CUID is passed to the host and device compilations of a
// translation unit, for both the old (--no-offload-new-driver) and new
// (--offload-new-driver) offloading models.

// DEFINE: %{clang_install_dir} = %/S/Inputs/SYCL/bin

// Default (-fuse-cuid=hash). Each translation unit gets its own CUID, shared
// by its host and device compilations.
// RUN: %clangxx -### -fsycl --no-offload-new-driver \
// RUN:   --target=x86_64-unknown-linux-gnu -c %s -x c++ \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu 2>&1 \
// RUN:   | FileCheck %s --check-prefix=HASH --implicit-check-not=warning:
// RUN: %clangxx -### -fsycl --offload-new-driver \
// RUN:   -ccc-install-dir %{clang_install_dir} \
// RUN:   --target=x86_64-unknown-linux-gnu -c %s -x c++ \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu 2>&1 \
// RUN:   | FileCheck %s --check-prefix=HASH --implicit-check-not=warning:
// RUN: %clangxx -### -fsycl --no-offload-new-driver -fuse-cuid=hash \
// RUN:   --target=x86_64-unknown-linux-gnu -c %s -x c++ \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu 2>&1 \
// RUN:   | FileCheck %s --check-prefix=HASH --implicit-check-not=warning:
// RUN: %clangxx -### -fsycl --offload-new-driver -fuse-cuid=hash \
// RUN:   -ccc-install-dir %{clang_install_dir} \
// RUN:   --target=x86_64-unknown-linux-gnu -c %s -x c++ \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu 2>&1 \
// RUN:   | FileCheck %s --check-prefix=HASH --implicit-check-not=warning:
// HASH: "-cc1" "-triple" "spir64-unknown-unknown"{{.*}} "-cuid=[[CUID:[0-9a-f]+]]"{{.*}} "{{.*}}sycl-cuid.cpp"
// HASH: "-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}} "-cuid=[[CUID]]"{{.*}} "{{.*}}sycl-cuid.cpp"
// HASH: "-cc1" "-triple" "spir64-unknown-unknown"
// HASH-NOT: "-cuid=[[CUID]]"
// HASH-SAME: "-cuid=[[CUID2:[0-9a-f]+]]"{{.*}} "{{.*}}a.cu"
// HASH: "-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}} "-cuid=[[CUID2]]"{{.*}} "{{.*}}a.cu"

// Windows (clang-cl) default hash.
// RUN: %clang_cl -### -fsycl --no-offload-new-driver \
// RUN:   --target=x86_64-pc-windows-msvc /c -- %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=HASH-WIN --implicit-check-not=warning:
// RUN: %clang_cl -### -fsycl --offload-new-driver \
// RUN:   -ccc-install-dir %{clang_install_dir} \
// RUN:   --target=x86_64-pc-windows-msvc /c -- %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=HASH-WIN --implicit-check-not=warning:
// HASH-WIN: "-cc1" "-triple" "spir64-unknown-unknown"{{.*}} "-cuid=[[CUID:[0-9a-f]+]]"
// HASH-WIN: "-cc1" "-triple" "x86_64-pc-windows-msvc{{.*}}"{{.*}} "-cuid=[[CUID]]"

// Explicit -cuid=<value> overrides -fuse-cuid and is used for all
// compilations, including Windows (clang-cl).
// RUN: %clangxx -### -fsycl --no-offload-new-driver -fuse-cuid=random \
// RUN:   -cuid=UNIQUE_anon1 \
// RUN:   --target=x86_64-unknown-linux-gnu -c %s 2>&1 \
// RUN:   | FileCheck %s -check-prefixes=FIXED,FIXED-LIN \
// RUN:     --implicit-check-not=warning:
// RUN: %clangxx -### -fsycl --offload-new-driver -fuse-cuid=random \
// RUN:   -cuid=UNIQUE_anon1 -ccc-install-dir %{clang_install_dir} \
// RUN:   --target=x86_64-unknown-linux-gnu -c %s 2>&1 \
// RUN:   | FileCheck %s -check-prefixes=FIXED,FIXED-LIN \
// RUN:     --implicit-check-not=warning:
// RUN: %clang_cl -### -fsycl --no-offload-new-driver \
// RUN:   --target=x86_64-pc-windows-msvc /clang:-cuid=UNIQUE_anon1 /c -- %s \
// RUN:   2>&1 | FileCheck %s -check-prefixes=FIXED,FIXED-WIN \
// RUN:     --implicit-check-not=warning:
// RUN: %clang_cl -### -fsycl --offload-new-driver \
// RUN:   -ccc-install-dir %{clang_install_dir} \
// RUN:   --target=x86_64-pc-windows-msvc /clang:-cuid=UNIQUE_anon1 /c -- %s \
// RUN:   2>&1 | FileCheck %s -check-prefixes=FIXED,FIXED-WIN \
// RUN:     --implicit-check-not=warning:
// FIXED: "-cc1" "-triple" "spir64-unknown-unknown"{{.*}} "-cuid=UNIQUE_anon1"
// FIXED-LIN: "-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}} "-cuid=UNIQUE_anon1"
// FIXED-WIN: "-cc1" "-triple" "x86_64-pc-windows-msvc{{.*}}"{{.*}} "-cuid=UNIQUE_anon1"

// -fuse-cuid=random: a single random CUID is shared by host and device.
// RUN: %clangxx -### -fsycl --no-offload-new-driver -fuse-cuid=random \
// RUN:   --target=x86_64-unknown-linux-gnu -c %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=RANDOM
// RUN: %clangxx -### -fsycl --offload-new-driver -fuse-cuid=random \
// RUN:   -ccc-install-dir %{clang_install_dir} \
// RUN:   --target=x86_64-unknown-linux-gnu -c %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=RANDOM
// RANDOM: "-cc1" "-triple" "spir64-unknown-unknown"{{.*}} "-cuid=[[CUID:[0-9a-f]+]]"
// RANDOM: "-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}} "-cuid=[[CUID]]"

// -fuse-cuid=none disables the CUID.
// RUN: %clangxx -### -fsycl --no-offload-new-driver -fuse-cuid=none \
// RUN:   --target=x86_64-unknown-linux-gnu -c %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NONE
// RUN: %clangxx -### -fsycl --offload-new-driver -fuse-cuid=none \
// RUN:   -ccc-install-dir %{clang_install_dir} \
// RUN:   --target=x86_64-unknown-linux-gnu -c %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NONE
// NONE-NOT: "-cuid=

// Multiple device targets share the CUID of the translation unit.
// RUN: %clangxx -### -fsycl --no-offload-new-driver \
// RUN:   -fsycl-targets=spir64,spir64_x86_64 \
// RUN:   --target=x86_64-unknown-linux-gnu -c %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=MULTI
// RUN: %clangxx -### -fsycl --offload-new-driver \
// RUN:   -fsycl-targets=spir64,spir64_x86_64 \
// RUN:   -ccc-install-dir %{clang_install_dir} \
// RUN:   --target=x86_64-unknown-linux-gnu -c %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=MULTI
// MULTI-DAG: "-cc1" "-triple" "spir64-unknown-unknown"{{.*}} "-cuid=[[CUID:[0-9a-f]+]]"
// MULTI-DAG: "-cc1" "-triple" "spir64_x86_64-unknown-unknown"{{.*}} "-cuid=[[CUID]]"
// MULTI-DAG: "-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}} "-cuid=[[CUID]]"

// -save-temps: preprocessing and compilation steps use the same CUID.
// RUN: %clangxx -### -fsycl --no-offload-new-driver -save-temps \
// RUN:   --target=x86_64-unknown-linux-gnu -c %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SAVE
// RUN: %clangxx -### -fsycl --offload-new-driver -save-temps \
// RUN:   -ccc-install-dir %{clang_install_dir} \
// RUN:   --target=x86_64-unknown-linux-gnu -c %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SAVE
// SAVE: "-cc1" "-triple" "spir64-unknown-unknown"{{.*}} "-cuid=[[CUID:[0-9a-f]+]]"{{.*}} "{{.*}}sycl-cuid.cpp"
// SAVE: "-cc1" "-triple" "spir64-unknown-unknown"{{.*}} "-cuid=[[CUID]]"{{.*}} "{{.*}}sycl-cuid-sycl-spir64-unknown-unknown.ii"
// SAVE: "-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}} "-cuid=[[CUID]]"{{.*}} "{{.*}}sycl-cuid.cpp"
// SAVE: "-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}} "-cuid=[[CUID]]"{{.*}} "{{.*}}sycl-cuid-host-x86_64-unknown-linux-gnu.ii"

// -fsycl-link of a source file.
// RUN: %clangxx -### -fsycl --no-offload-new-driver -fsycl-link \
// RUN:   -ccc-install-dir %{clang_install_dir} \
// RUN:   --target=x86_64-unknown-linux-gnu %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=LINK
// RUN: %clangxx -### -fsycl --offload-new-driver -fsycl-link \
// RUN:   -ccc-install-dir %{clang_install_dir} \
// RUN:   --target=x86_64-unknown-linux-gnu %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=LINK
// LINK: "-cc1" "-triple" "spir64-unknown-unknown"{{.*}} "-cuid=[[CUID:[0-9a-f]+]]"{{.*}} "{{.*}}sycl-cuid.cpp"
// LINK: "-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}} "-cuid=[[CUID]]"

// Third party host compiler: the device compilation (which generates the
// integration header) receives the CUID; the host compiler does not.
// RUN: %clangxx -### -fsycl --no-offload-new-driver -fsycl-host-compiler=g++ \
// RUN:   --target=x86_64-unknown-linux-gnu -c %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=HOSTCOMP --implicit-check-not=warning:
// RUN: %clangxx -### -fsycl --offload-new-driver -fsycl-host-compiler=g++ \
// RUN:   -ccc-install-dir %{clang_install_dir} \
// RUN:   --target=x86_64-unknown-linux-gnu -c %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=HOSTCOMP --implicit-check-not=warning:
// HOSTCOMP: "-cc1" "-triple" "spir64-unknown-unknown"{{.*}} "-fsycl-int-header={{.*}} "-cuid={{[0-9a-f]+}}"
// HOSTCOMP: "{{.*}}g++" "{{.*}}.cpp" "-c" "-include"
// HOSTCOMP-NOT: -cuid=

// Invalid -fuse-cuid value.
// RUN: not %clangxx -### -fsycl --no-offload-new-driver -fuse-cuid=bad \
// RUN:   --target=x86_64-unknown-linux-gnu -c %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=INVALID
// RUN: not %clangxx -### -fsycl --offload-new-driver -fuse-cuid=bad \
// RUN:   -ccc-install-dir %{clang_install_dir} \
// RUN:   --target=x86_64-unknown-linux-gnu -c %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=INVALID
// INVALID: error: invalid value 'bad' in '-fuse-cuid=bad'
