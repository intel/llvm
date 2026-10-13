/// Test that -fsycl-allow-device-image-dependencies on Windows force-loads
/// user import libs by passing /INCLUDE:__imp_<sym>, across every way a .lib
/// can be named and located.
///
/// The library-name form (direct filename, /link, /defaultlib:, /wholearchive:,
/// -l, -Wl,/-Xlinker, @response-file) is crossed with how the .lib is found
/// (a full path, /libpath:, the LIB env var, or -L).

// REQUIRES: system-windows

/// Build a minimal "dep.lib" in its own directory so the directory can be
/// handed to the linker search path. The driver only scans the archive's
/// symbol table for a name starting with "__imp_", so a plain object defining
/// __imp_TestFunc, archived with llvm-ar, is enough — no real import lib
/// required.
// RUN: rm -rf %t.libdir && mkdir -p %t.libdir
// RUN: echo ".globl __imp_TestFunc" > %t.s
// RUN: echo ".data" >> %t.s
// RUN: echo "__imp_TestFunc:" >> %t.s
// RUN: echo ".quad 0" >> %t.s
// RUN: llvm-mc -triple=x86_64-windows-msvc -filetype=obj -o %t.o %t.s
// RUN: llvm-ar crs %t.libdir/dep.lib %t.o

/// ---------------------------------------------------------------------------
/// clang-cl driver: the library reaches link.exe as a cl-style argument.
/// ---------------------------------------------------------------------------

/// (1) Full path named directly as an input.
// RUN: %clang_cl -fsycl /clang:--sysroot=%S/Inputs/SYCL \
// RUN:          -fsycl-allow-device-image-dependencies /O2 -### %s \
// RUN:          %t.libdir/dep.lib 2>&1 | FileCheck %s

/// (2) Full path passed through /link.
// RUN: %clang_cl -fsycl /clang:--sysroot=%S/Inputs/SYCL \
// RUN:          -fsycl-allow-device-image-dependencies /O2 -### %s \
// RUN:          /link %t.libdir/dep.lib 2>&1 | FileCheck %s

/// (3) Bare name via /link, resolved through /libpath:.
// RUN: %clang_cl -fsycl /clang:--sysroot=%S/Inputs/SYCL \
// RUN:          -fsycl-allow-device-image-dependencies /O2 -### %s \
// RUN:          /link dep.lib /libpath:%t.libdir 2>&1 | FileCheck %s

/// (4) Bare name via /link, resolved through the LIB env var.
// RUN: env "LIB=%t.libdir" %clang_cl -fsycl /clang:--sysroot=%S/Inputs/SYCL \
// RUN:          -fsycl-allow-device-image-dependencies /O2 -### %s \
// RUN:          /link dep.lib 2>&1 | FileCheck %s

/// (5) /defaultlib: with the .lib extension omitted (the driver appends it),
///     resolved through /libpath:.
// RUN: %clang_cl -fsycl /clang:--sysroot=%S/Inputs/SYCL \
// RUN:          -fsycl-allow-device-image-dependencies /O2 -### %s \
// RUN:          /link /defaultlib:dep /libpath:%t.libdir 2>&1 | FileCheck %s

/// (6) /wholearchive: with a bare name, resolved through /libpath:.
// RUN: %clang_cl -fsycl /clang:--sysroot=%S/Inputs/SYCL \
// RUN:          -fsycl-allow-device-image-dependencies /O2 -### %s \
// RUN:          /link /wholearchive:dep /libpath:%t.libdir 2>&1 | FileCheck %s

/// (7) A .lib pulled in via a linker response file (bare name + /libpath:).
// RUN: echo "dep.lib /libpath:%t.libdir" > %t.rsp
// RUN: %clang_cl -fsycl /clang:--sysroot=%S/Inputs/SYCL \
// RUN:          -fsycl-allow-device-image-dependencies /O2 -### %s \
// RUN:          /link @%t.rsp 2>&1 | FileCheck %s

/// ---------------------------------------------------------------------------
/// clang driver (GNU-style): -l / -L / -Wl / -Xlinker forms.
/// ---------------------------------------------------------------------------

/// (8) -l<name> resolved through -L.
// RUN: %clang -fsycl --sysroot=%S/Inputs/SYCL \
// RUN:          -fsycl-allow-device-image-dependencies -O2 -### %s \
// RUN:          -L%t.libdir -ldep 2>&1 | FileCheck %s

/// (9) Full path forwarded verbatim via -Wl,.
// RUN: %clang -fsycl --sysroot=%S/Inputs/SYCL \
// RUN:          -fsycl-allow-device-image-dependencies -O2 -### %s \
// RUN:          -Wl,%t.libdir/dep.lib 2>&1 | FileCheck %s

/// (10) Full path forwarded verbatim via -Xlinker.
// RUN: %clang -fsycl --sysroot=%S/Inputs/SYCL \
// RUN:          -fsycl-allow-device-image-dependencies -O2 -### %s \
// RUN:          -Xlinker %t.libdir/dep.lib 2>&1 | FileCheck %s

/// (11) A .lib named only by a /DEFAULTLIB: directive embedded in an object
///      file's COFF .drectve section (as /Qmkl and #pragma comment(lib) do),
///      never on the command line. Resolved through /libpath:.
// RUN: echo '#pragma comment(lib, "dep.lib")' > %t.dep.c
// RUN: %clang_cl /clang:--sysroot=%S/Inputs/SYCL -c -o %t.dep.obj %t.dep.c
// RUN: %clang_cl -fsycl /clang:--sysroot=%S/Inputs/SYCL \
// RUN:          -fsycl-allow-device-image-dependencies /O2 -### %s \
// RUN:          %t.dep.obj /link /libpath:%t.libdir 2>&1 | FileCheck %s

/// (12) Same, but the object carrying the directive is a *fat* -fsycl object.
///      The driver hands the linker an unbundled temp (which does not exist at
///      -### time), so the directive must be read from the original on-disk
///      base input, not the linker's filename. Guards the getBaseInput() switch
///      that (11), a plain object where the two names coincide, cannot.
// RUN: %clang_cl -fsycl /clang:--sysroot=%S/Inputs/SYCL -c -o %t.fatdep.obj %t.dep.c
// RUN: %clang_cl -fsycl /clang:--sysroot=%S/Inputs/SYCL \
// RUN:          -fsycl-allow-device-image-dependencies /O2 -### %s \
// RUN:          %t.fatdep.obj /link /libpath:%t.libdir 2>&1 | FileCheck %s

// CHECK: "/INCLUDE:__imp_TestFunc"

/// ---------------------------------------------------------------------------
/// Negative: a named system library (kernel32.lib, found on the LIB env var
/// the toolchain sets up) must never be force-loaded.
/// ---------------------------------------------------------------------------
// RUN: %clang_cl -fsycl /clang:--sysroot=%S/Inputs/SYCL \
// RUN:          -fsycl-allow-device-image-dependencies /O2 -### %s \
// RUN:          /link kernel32.lib 2>&1 | FileCheck --check-prefix=SYSLIB %s
/// The linker line is emitted (driver did not crash scanning the .lib) but
/// carries no force-load directive.
// SYSLIB: link.exe
// SYSLIB-NOT: "/INCLUDE:__imp_

/// ---------------------------------------------------------------------------
/// Negative gating: force-loading only kicks in when *both* -fsycl and
/// -fsycl-allow-device-image-dependencies are present. Use the resolvable
/// dep.lib so only the missing flag can suppress the /INCLUDE:.
/// ---------------------------------------------------------------------------

/// -fsycl-allow-device-image-dependencies without -fsycl.
// RUN: %clang_cl /clang:--sysroot=%S/Inputs/SYCL \
// RUN:          -fsycl-allow-device-image-dependencies /O2 -### %s \
// RUN:          %t.libdir/dep.lib 2>&1 | FileCheck --check-prefix=NONE %s

/// -fsycl without -fsycl-allow-device-image-dependencies.
// RUN: %clang_cl -fsycl /clang:--sysroot=%S/Inputs/SYCL \
// RUN:          /O2 -### %s \
// RUN:          %t.libdir/dep.lib 2>&1 | FileCheck --check-prefix=NONE %s

/// ---------------------------------------------------------------------------
/// Negative: a .lib that cannot be resolved emits no /INCLUDE: (and the driver
/// does not fail). Covers every resolution path — a bare name with no search
/// dir, a name that misses in an existing /libpath:, a nonexistent full path,
/// the extension-appending /defaultlib:/wholearchive: forms, and a missing
/// @response-file.
/// ---------------------------------------------------------------------------

/// Bare name, nothing on the search path.
// RUN: %clang_cl -fsycl /clang:--sysroot=%S/Inputs/SYCL \
// RUN:          -fsycl-allow-device-image-dependencies /O2 -### %s \
// RUN:          /link nosuch.lib 2>&1 | FileCheck --check-prefix=NONE %s

/// Name misses in a /libpath: dir that exists but lacks it.
// RUN: %clang_cl -fsycl /clang:--sysroot=%S/Inputs/SYCL \
// RUN:          -fsycl-allow-device-image-dependencies /O2 -### %s \
// RUN:          /link nosuch.lib /libpath:%t.libdir 2>&1 | FileCheck --check-prefix=NONE %s

/// Nonexistent full path named directly.
// RUN: %clang_cl -fsycl /clang:--sysroot=%S/Inputs/SYCL \
// RUN:          -fsycl-allow-device-image-dependencies /O2 -### %s \
// RUN:          %t.libdir/nosuch.lib 2>&1 | FileCheck --check-prefix=NONE %s

/// /defaultlib: (extension appended) that resolves to nothing.
// RUN: %clang_cl -fsycl /clang:--sysroot=%S/Inputs/SYCL \
// RUN:          -fsycl-allow-device-image-dependencies /O2 -### %s \
// RUN:          /link /defaultlib:nosuch /libpath:%t.libdir 2>&1 | FileCheck --check-prefix=NONE %s

/// /wholearchive: that resolves to nothing.
// RUN: %clang_cl -fsycl /clang:--sysroot=%S/Inputs/SYCL \
// RUN:          -fsycl-allow-device-image-dependencies /O2 -### %s \
// RUN:          /link /wholearchive:nosuch.lib 2>&1 | FileCheck --check-prefix=NONE %s

/// Missing response file.
// RUN: %clang_cl -fsycl /clang:--sysroot=%S/Inputs/SYCL \
// RUN:          -fsycl-allow-device-image-dependencies /O2 -### %s \
// RUN:          /link @%t.nosuch.rsp 2>&1 | FileCheck --check-prefix=NONE %s

/// The linker line is emitted (driver did not crash resolving the .lib) but
/// carries no force-load directive.
// NONE: link.exe
// NONE-NOT: "/INCLUDE:__imp_
