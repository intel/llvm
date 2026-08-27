///
/// Tests for --ocloc-path=, which provides the location of the externally
/// acquired ocloc tool used for Intel GPU AOT compilation.
///

// REQUIRES: x86-registered-target

/// Check that --ocloc-path= is used for the old offloading model.
// RUN:   %clang -### -fsycl --no-offload-new-driver -fsycl-targets=spir64_gen \
// RUN:     --ocloc-path=/my/ocloc/dir %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-OCLOC-PATH-OLD %s
// RUN:   %clang -### -fsycl --no-offload-new-driver \
// RUN:     -fsycl-targets=intel_gpu_pvc --ocloc-path=/my/ocloc/dir %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-OCLOC-PATH-OLD %s
// CHK-OCLOC-PATH-OLD: "/my/ocloc/dir{{[/\\]+}}ocloc{{(\.exe)?}}" "-output"

/// Check that the user provided location wins over an ocloc that is visible
/// via the PATH.  The fake ocloc must be findable, which means it needs the
/// execute bit set on linux and the executable extension on windows.
// RUN:   rm -rf %t.dir && mkdir -p %t.dir
// RUN:   %if system-windows %{ touch %t.dir/ocloc.exe %} \
// RUN:   %else %{ touch %t.dir/ocloc && chmod +x %t.dir/ocloc %}
// RUN:   env "PATH=%t.dir%{pathsep}%PATH%" %clang -### -fsycl \
// RUN:     --no-offload-new-driver -fsycl-targets=spir64_gen \
// RUN:     --ocloc-path=/my/ocloc/dir %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-OCLOC-PATH-OLD %s

/// Check that the 'exe' name is used for windows.
// RUN:   %clang_cl -### -fsycl --no-offload-new-driver \
// RUN:     -fsycl-targets=spir64_gen --ocloc-path=/my/ocloc/dir -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-OCLOC-PATH-OLD-WIN %s
// RUN:   %clang -### -target x86_64-pc-windows-msvc -fsycl \
// RUN:     --no-offload-new-driver -fsycl-targets=spir64_gen \
// RUN:     --ocloc-path=/my/ocloc/dir %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-OCLOC-PATH-OLD-WIN %s
// CHK-OCLOC-PATH-OLD-WIN: "/my/ocloc/dir{{[/\\]+}}ocloc.exe" "-output"

/// Check that --ocloc-path= is forwarded to the clang-linker-wrapper for the
/// new offloading model.
// RUN:   %clang -### -fsycl --offload-new-driver -fsycl-targets=spir64_gen \
// RUN:     --sysroot=%S/Inputs/SYCL --ocloc-path=/my/ocloc/dir %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-OCLOC-PATH-NEW %s
// RUN:   %clang -### -fsycl --offload-new-driver \
// RUN:     -fsycl-targets=intel_gpu_pvc --sysroot=%S/Inputs/SYCL \
// RUN:     --ocloc-path=/my/ocloc/dir %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-OCLOC-PATH-NEW %s
// CHK-OCLOC-PATH-NEW: clang-linker-wrapper{{.*}} "--ocloc-path=/my/ocloc/dir"

/// Check that --ocloc-path= is forwarded to the clang-sycl-linker.
// RUN:   touch %t.bc
// RUN:   %clangxx -### --target=spirv64 --sycl-link \
// RUN:     --ocloc-path=/my/ocloc/dir %t.bc 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-OCLOC-PATH-SYCL-LINK %s
// CHK-OCLOC-PATH-SYCL-LINK: clang-sycl-linker{{.*}} "--ocloc-path=/my/ocloc/dir"

/// Check that --ocloc-path= is used when emitting the ocloc help information.
// RUN:   %clang -### -fsycl -fsycl-help=gen --ocloc-path=/my/ocloc/dir %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-OCLOC-PATH-HELP %s
// CHK-OCLOC-PATH-HELP: Emitting help information for ocloc
// CHK-OCLOC-PATH-HELP: "/my/ocloc/dir{{[/\\]+}}ocloc{{(\.exe)?}}" "--help"

/// Check that the 'exe' name is used for windows when emitting the ocloc help
/// information.
// RUN:   %clang -### -target x86_64-pc-windows-msvc -fsycl -fsycl-help=gen \
// RUN:     --ocloc-path=/my/ocloc/dir %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-OCLOC-PATH-HELP-WIN %s
// CHK-OCLOC-PATH-HELP-WIN: "/my/ocloc/dir{{[/\\]+}}ocloc.exe" "--help"

/// Check the diagnostic emitted when the given directory does not contain a
/// usable ocloc.  Without -### the tool is actually launched, so the composed
/// path is expected to be diagnosed instead of being silently ignored.
// RUN:   rm -rf %t.empty.dir && mkdir -p %t.empty.dir
// RUN:   not %clang -fsycl -fsycl-help=gen --ocloc-path=%t.empty.dir %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-OCLOC-PATH-HELP-ERR %s
// CHK-OCLOC-PATH-HELP-ERR: error: unable to execute command: {{.*}}ocloc

/// Check that an empty --ocloc-path= is rejected instead of falling back to
/// another ocloc.
// RUN:   not %clang -### -fsycl --no-offload-new-driver \
// RUN:     -fsycl-targets=spir64_gen --ocloc-path= %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-OCLOC-PATH-EMPTY %s
// RUN:   not %clang -fsycl -fsycl-help=gen --ocloc-path= %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-OCLOC-PATH-EMPTY %s
// CHK-OCLOC-PATH-EMPTY: error: invalid value '' in '--ocloc-path='
// CHK-OCLOC-PATH-EMPTY-NOT: Emitting help information

/// Check that --ocloc-path= does not warn as unused when no AOT compilation
/// for Intel GPU is being performed.
// RUN:   %clang -### -fsycl -fsycl-targets=spir64 --ocloc-path=/my/ocloc/dir \
// RUN:     --sysroot=%S/Inputs/SYCL %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-OCLOC-PATH-UNUSED %s
// CHK-OCLOC-PATH-UNUSED-NOT: warning: argument unused during compilation

/// Check that a --ocloc-path= directory whose name contains spaces is
/// composed and quoted correctly.
// RUN:   %clang -### -fsycl --no-offload-new-driver -fsycl-targets=spir64_gen \
// RUN:     --ocloc-path="/my/ocloc dir/with spaces" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-OCLOC-PATH-SPACES-OLD %s
// CHK-OCLOC-PATH-SPACES-OLD: "/my/ocloc dir/with spaces{{[/\\]+}}ocloc{{(\.exe)?}}" "-output"
// RUN:   %clang -### -fsycl --offload-new-driver -fsycl-targets=spir64_gen \
// RUN:     --sysroot=%S/Inputs/SYCL --ocloc-path="/my/ocloc dir/with spaces" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-OCLOC-PATH-SPACES-NEW %s
// CHK-OCLOC-PATH-SPACES-NEW: clang-linker-wrapper{{.*}} "--ocloc-path=/my/ocloc dir/with spaces"
// RUN:   %clangxx -### --target=spirv64 --sycl-link \
// RUN:     --ocloc-path="/my/ocloc dir/with spaces" %t.bc 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-OCLOC-PATH-SPACES-SYCL-LINK %s
// CHK-OCLOC-PATH-SPACES-SYCL-LINK: clang-sycl-linker{{.*}} "--ocloc-path=/my/ocloc dir/with spaces"

/// Check that a real --ocloc-path= directory containing spaces is actually
/// located and used to launch the tool.
// RUN:   rm -rf "%t.dir with spaces" && mkdir -p "%t.dir with spaces"
// RUN:   not %clang -fsycl -fsycl-help=gen \
// RUN:     --ocloc-path="%t.dir with spaces" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-OCLOC-PATH-SPACES-ERR %s
// CHK-OCLOC-PATH-SPACES-ERR: error: unable to execute command: {{.*}}dir with spaces{{[/\\]+}}ocloc
