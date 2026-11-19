// The MD5-derived hash of `-cuid=abc` (see ASTContext::getCUIDHash):
// DEFINE: %{hash} = b04fd23c98500190
//
// RUN: %clang_cc1 -fsycl-is-device -fsycl-unnamed-lambda \
// RUN:   -internal-isystem %S/Inputs -triple spir64-unknown-unknown -cuid=abc \
// RUN:   -fsycl-int-header=%t.h -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=DEVICE -DHASH=%{hash}
// RUN: FileCheck %s --input-file=%t.h --check-prefix=INT-HEADER -DHASH=%{hash}
// RUN: %clang_cc1 -fsycl-is-host -fsycl-unnamed-lambda \
// RUN:   -internal-isystem %S/Inputs -triple x86_64-unknown-linux-gnu \
// RUN:   -cuid=abc -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=HOST -DHASH=%{hash}
// RUN: %clang_cc1 -fsycl-is-device -fsycl-unnamed-lambda \
// RUN:   -internal-isystem %S/Inputs -triple spir64-unknown-unknown \
// RUN:   -fsycl-int-header=%t.nocuid.h -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=NOCUID --implicit-check-not=cuid_
// RUN: FileCheck %s --input-file=%t.nocuid.h --check-prefix=NOCUID \
// RUN:   --implicit-check-not=cuid_
// RUN: %clang_cc1 -fsycl-is-host -fsycl-unnamed-lambda \
// RUN:   -internal-isystem %S/Inputs -triple x86_64-unknown-linux-gnu \
// RUN:   -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=HOST-NOCUID --implicit-check-not=cuid_
//
// The driver never selects triples asking for Microsoft C++ ABI, but such
// triples are still accepted when given explicitly. In this case, CUID is added
// as a synthetic outermost scope instead, instead of leaving TU-local kernel
// names to collide.
// RUN: %clang_cc1 -fsycl-is-device -fsycl-unnamed-lambda \
// RUN:   -internal-isystem %S/Inputs -triple spir64-pc-windows-msvc \
// RUN:   -aux-triple x86_64-pc-windows-msvc -cuid=abc \
// RUN:   -fsycl-int-header=%t.msvc.h -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=MSVC -DHASH=%{hash}
// RUN: FileCheck %s --input-file=%t.msvc.h --check-prefix=MSVC-INT-HEADER \
// RUN:   -DHASH=%{hash}
// RUN: %clang_cc1 -fsycl-is-device -fsycl-unnamed-lambda \
// RUN:   -internal-isystem %S/Inputs -triple spir64-pc-windows-msvc \
// RUN:   -aux-triple x86_64-pc-windows-msvc -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=MSVC-NOCUID --implicit-check-not=cuid_

// Verify that with `-cuid`, mangled names of TU-local kernels have a per-TU
// compilation unit ID inserted into them, to prevent TU-local kernels with
// identical mangled names from silently merging at device link time.

#include "sycl.hpp"

// Used here to expose strings computed by __builtin_sycl_unique_stable_name:
extern "C" int puts(const char *);

static void staticUnnamed(sycl::queue &q) {
  // Unnamed kernel in a function with internal linkage:
  auto k = [] {};
  q.submit([&](sycl::handler &h) { h.single_task(k); });
  // Compute k's kernel name as SYCL headers would do on host, in order to
  // find the kernel by name later:
  puts(__builtin_sycl_unique_stable_name(decltype(k)));
  // Host, device, and integration header all need to agree:
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSU21cuid_[[HASH]]ZL13staticUnnamedRN4sycl3_V15queueEEUlvE_(
  // INT-HEADER-DAG: getName() { return "_ZTSU21cuid_[[HASH]]ZL13staticUnnamedRN4sycl3_V15queueEEUlvE_"; }
  // INT-HEADER-DAG: struct KernelInfoData<'_', 'Z', 'T', 'S', 'U', '2', '1', 'c', 'u', 'i', 'd', '_',
  // HOST-DAG: c"_ZTSU21cuid_[[HASH]]ZL13staticUnnamedRN4sycl3_V15queueEEUlvE_\00"
  // HOST-NOCUID-DAG: c"_ZTSZL13staticUnnamedRN4sycl3_V15queueEEUlvE_\00"
}

void extUnnamed(sycl::queue &q) {
  // Unnamed kernel in an externally visible function, don't add CUID:
  auto k = [] {};
  q.submit([&](sycl::handler &h) { h.single_task(k); });
  puts(__builtin_sycl_unique_stable_name(decltype(k)));
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSZ10extUnnamedRN4sycl3_V15queueEEUlvE_(
  // INT-HEADER-DAG: getName() { return "_ZTSZ10extUnnamedRN4sycl3_V15queueEEUlvE_"; }
  // HOST-DAG: c"_ZTSZ10extUnnamedRN4sycl3_V15queueEEUlvE_\00"

  // Unnamed kernel nested in a command group lambda: even if it's nested
  // deeper, since extUnnamed is still externally visible, we don't need a CUID
  // here either:
  q.submit([&](sycl::handler &h) { h.single_task([] {}); });
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSZZ10extUnnamedRN4sycl3_V15queueEENKUlRNS0_7handlerEE0_clES4_EUlvE_(
  // INT-HEADER-DAG: getName() { return "_ZTSZZ10extUnnamedRN4sycl3_V15queueEENKUlRNS0_7handlerEE0_clES4_EUlvE_"; }
  // No HOST check here: the nested lambda's type can only be named inside the
  // command group, which the mock queue::submit never invokes, so nothing
  // inside it is emitted on host.

}

inline void inlineFn(sycl::queue &q) {
  // Unnamed kernel in an inline function, which should be the same kernel in
  // every TU that includes it, so don't add CUID:
  auto k = [] {};
  q.submit([&](sycl::handler &h) { h.single_task(k); });
  puts(__builtin_sycl_unique_stable_name(decltype(k)));
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSZ8inlineFnRN4sycl3_V15queueEEUlvE_(
  // INT-HEADER-DAG: getName() { return "_ZTSZ8inlineFnRN4sycl3_V15queueEEUlvE_"; }
  // HOST-DAG: c"_ZTSZ8inlineFnRN4sycl3_V15queueEEUlvE_\00"
}

namespace {
struct AnonFunc { void operator()() const {} };
} // namespace

struct ExtFunc { void operator()() const {} };

void funcObjects(sycl::queue &q) {
  // Unnamed kernel given as a function object whose type is in an unnamed
  // namespace:
  q.submit([&](sycl::handler &h) { h.single_task(AnonFunc{}); });
  puts(__builtin_sycl_unique_stable_name(AnonFunc));
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSU21cuid_[[HASH]]N12_GLOBAL__N_18AnonFuncE(
  // INT-HEADER-DAG: getName() { return "_ZTSU21cuid_[[HASH]]N12_GLOBAL__N_18AnonFuncE"; }
  // HOST-DAG: c"_ZTSU21cuid_[[HASH]]N12_GLOBAL__N_18AnonFuncE\00"
  // NOCUID-DAG: _ZTSN12_GLOBAL__N_18AnonFuncE

  // This time the function object is externally visible, so don't add CUID:
  q.submit([&](sycl::handler &h) { h.single_task(ExtFunc{}); });
  puts(__builtin_sycl_unique_stable_name(ExtFunc));
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTS7ExtFunc(
  // INT-HEADER-DAG: getName() { return "_ZTS7ExtFunc"; }
  // HOST-DAG: c"_ZTS7ExtFunc\00"
}

// Kernel names declared in place are local to the function that submits the
// kernel, so they are only TU-local when their parent functions are.
static void staticNamed(sycl::queue &q) {
  // Enclosing function has internal linkage:
  q.submit([&](sycl::handler &h) { h.single_task<class K>([] {}); });
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSU21cuid_[[HASH]]ZZL11staticNamedRN4sycl3_V15queueEENKUlRNS0_7handlerEE_clES4_E1K(
  // INT-HEADER-DAG: getName() { return "_ZTSU21cuid_[[HASH]]ZZL11staticNamedRN4sycl3_V15queueEENKUlRNS0_7handlerEE_clES4_E1K"; }
  // NOCUID-DAG: _ZTSZZL11staticNamedRN4sycl3_V15queueEENKUlRNS0_7handlerEE_clES4_E1K
  // MSVC-DAG: define {{.*}}spir_kernel void @"?AVK@?1???R<lambda_1>@?0??staticNamed@@YAXAEAVqueue@_V1@sycl@@@Z@QEBA?A?<auto>@@AEAVhandler@45@@Z<cuid_[[HASH]]>@@"(
  // MSVC-INT-HEADER-DAG: getName() { return "?AVK@?1???R<lambda_1>@?0??staticNamed@@YAXAEAVqueue@_V1@sycl@@@Z@QEBA?A?<auto>@@AEAVhandler@45@@Z<cuid_[[HASH]]>@@"; }
  // MSVC-NOCUID-DAG: define {{.*}}spir_kernel void @"?AVK@?1???R<lambda_1>@?0??staticNamed@@YAXAEAVqueue@_V1@sycl@@@Z@QEBA?A?<auto>@@AEAVhandler@45@@Z@"(
}

// Two distinct kernel names with the same name, local to the same function
// with internal linkage, are distinguished by a discriminator assigned by the
// mangler. The CUID must be added to these mangled names rather than falling
// back to the stable names, which lack the discriminator and would collide:
static void sameNamedLocals(sycl::queue &q) {
  q.submit([&](sycl::handler &h) {
    { h.single_task<class KLocal>([] {}); }
    { h.single_task<class KLocal>([] {}); }
  });
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSU21cuid_[[HASH]]ZZL15sameNamedLocalsRN4sycl3_V15queueEENKUlRNS0_7handlerEE_clES4_E6KLocal(
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSU21cuid_[[HASH]]ZZL15sameNamedLocalsRN4sycl3_V15queueEENKUlRNS0_7handlerEE_clES4_E6KLocal_0(
  // MSVC-DAG: define {{.*}}spir_kernel void @"?AVKLocal@?1???R<lambda_1>@?0??sameNamedLocals@@YAXAEAVqueue@_V1@sycl@@@Z@QEBA?A?<auto>@@AEAVhandler@45@@Z<cuid_[[HASH]]>@@"(
  // MSVC-DAG: define {{.*}}spir_kernel void @"?AVKLocal@?2???R<lambda_1>@?0??sameNamedLocals@@YAXAEAVqueue@_V1@sycl@@@Z@QEBA?A?<auto>@@AEAVhandler@45@@Z<cuid_[[HASH]]>@@"(
}

void extNamed(sycl::queue &q) {
  // Enclosing function is externally visible, so don't add CUID:
  q.submit([&](sycl::handler &h) { h.single_task<class K>([] {}); });
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSZZ8extNamedRN4sycl3_V15queueEENKUlRNS0_7handlerEE_clES4_E1K(
  // INT-HEADER-DAG: getName() { return "_ZTSZZ8extNamedRN4sycl3_V15queueEENKUlRNS0_7handlerEE_clES4_E1K"; }
  // MSVC-DAG: define {{.*}}spir_kernel void @"?AVK@?1???R<lambda_1>@?0??extNamed@@YAXAEAVqueue@_V1@sycl@@@Z@QEBA?A?<auto>@@AEAVhandler@45@@Z@"(
  // MSVC-INT-HEADER-DAG: getName() { return "?AVK@?1???R<lambda_1>@?0??extNamed@@YAXAEAVqueue@_V1@sycl@@@Z@QEBA?A?<auto>@@AEAVhandler@45@@Z@"; }
}

namespace {
struct Anonymous {};
} // namespace
struct External {};

template <typename T> class KTmpl;

template <typename T> void templateFn(sycl::queue &q) {
  auto k = [] {};
  q.submit([&](sycl::handler &h) { h.single_task(k); });
  puts(__builtin_sycl_unique_stable_name(decltype(k)));

  // If instantiated with a type in an unnamed namespace
  // (templateFn<Anonymous>), add CUID: Otherwise another TU doing the same with
  // its own 'Anonymous' would get the same name and get silently dropped at
  // device link time.
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSU21cuid_[[HASH]]Z10templateFnIN12_GLOBAL__N_19AnonymousEEvRN4sycl3_V15queueEEUlvE_(
  // INT-HEADER-DAG: getName() { return "_ZTSU21cuid_[[HASH]]Z10templateFnIN12_GLOBAL__N_19AnonymousEEvRN4sycl3_V15queueEEUlvE_"; }
  // HOST-DAG: c"_ZTSU21cuid_[[HASH]]Z10templateFnIN12_GLOBAL__N_19AnonymousEEvRN4sycl3_V15queueEEUlvE_\00"
  // Don't add CUID if we don't have -cuid:
  // NOCUID-DAG: _ZTSZ10templateFnIN12_GLOBAL__N_19AnonymousEEvRN4sycl3_V15queueEEUlvE_

  // The same template instantiated with an externally visible type
  // (templateFn<External>) is the same kernel in every TU, so don't add CUID:
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSZ10templateFnI8ExternalEvRN4sycl3_V15queueEEUlvE_(
  // INT-HEADER-DAG: getName() { return "_ZTSZ10templateFnI8ExternalEvRN4sycl3_V15queueEEUlvE_"; }
  // HOST-DAG: c"_ZTSZ10templateFnI8ExternalEvRN4sycl3_V15queueEEUlvE_\00"

  // Same for an explicit kernel name that is a template specialization:
  q.submit([&](sycl::handler &h) { h.single_task<KTmpl<T>>([] {}); });
  puts(__builtin_sycl_unique_stable_name(KTmpl<T>));

  // With templateFn<Anonymous>:
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSU21cuid_[[HASH]]5KTmplIN12_GLOBAL__N_19AnonymousEE(
  // INT-HEADER-DAG: getName() { return "_ZTSU21cuid_[[HASH]]5KTmplIN12_GLOBAL__N_19AnonymousEE"; }
  // HOST-DAG: c"_ZTSU21cuid_[[HASH]]5KTmplIN12_GLOBAL__N_19AnonymousEE\00"
  // NOCUID-DAG: _ZTS5KTmplIN12_GLOBAL__N_19AnonymousEE
  // HOST-NOCUID-DAG: c"_ZTS5KTmplIN12_GLOBAL__N_19AnonymousEE\00"
  // MSVC-DAG: define {{.*}}spir_kernel void @"?AV?$KTmpl@UAnonymous@?A0x{{[0-9A-F]+}}@@@<cuid_[[HASH]]>@@"(
  // MSVC-INT-HEADER-DAG: getName() { return "?AV?$KTmpl@UAnonymous@?A0x{{[0-9A-F]+}}@@@<cuid_[[HASH]]>@@"; }
  // MSVC-NOCUID-DAG: define {{.*}}spir_kernel void @"?AV?$KTmpl@UAnonymous@?A0x{{[0-9A-F]+}}@@@@"(

  // With templateFn<External>:
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTS5KTmplI8ExternalE(
  // INT-HEADER-DAG: getName() { return "_ZTS5KTmplI8ExternalE"; }
  // HOST-DAG: c"_ZTS5KTmplI8ExternalE\00"
  // MSVC-DAG: define {{.*}}spir_kernel void @"?AV?$KTmpl@UExternal@@@@"(
  // MSVC-INT-HEADER-DAG: getName() { return "?AV?$KTmpl@UExternal@@@@"; }
}

namespace {
class KAnon;
}
class KExt;

void explicitNames(sycl::queue &q) {
  // Named kernel, but name is in an unnamed namespace, so add CUID:
  q.submit([&](sycl::handler &h) { h.single_task<KAnon>([] {}); });
  puts(__builtin_sycl_unique_stable_name(KAnon));
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSU21cuid_[[HASH]]N12_GLOBAL__N_15KAnonE(
  // INT-HEADER-DAG: getName() { return "_ZTSU21cuid_[[HASH]]N12_GLOBAL__N_15KAnonE"; }
  // HOST-DAG: c"_ZTSU21cuid_[[HASH]]N12_GLOBAL__N_15KAnonE\00"
  // NOCUID-DAG: _ZTSN12_GLOBAL__N_15KAnonE
  // HOST-NOCUID-DAG: c"_ZTSN12_GLOBAL__N_15KAnonE\00"
  // MSVC-DAG: define {{.*}}spir_kernel void @"?AVKAnon@?A0x{{[0-9A-F]+}}@<cuid_[[HASH]]>@@"(
  // MSVC-INT-HEADER-DAG: getName() { return "?AVKAnon@?A0x{{[0-9A-F]+}}@<cuid_[[HASH]]>@@"; }
  // MSVC-NOCUID-DAG: define {{.*}}spir_kernel void @"?AVKAnon@?A0x{{[0-9A-F]+}}@@"(

  // Externally visible kernel name, don't add CUID:
  q.submit([&](sycl::handler &h) { h.single_task<KExt>([] {}); });
  puts(__builtin_sycl_unique_stable_name(KExt));
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTS4KExt(
  // INT-HEADER-DAG: getName() { return "_ZTS4KExt"; }
  // HOST-DAG: c"_ZTS4KExt\00"
  // MSVC-DAG: define {{.*}}spir_kernel void @"?AVKExt@@"(
  // MSVC-INT-HEADER-DAG: getName() { return "?AVKExt@@"; }
}

int main() {
  sycl::queue q;

  staticUnnamed(q);
  templateFn<Anonymous>(q);
  templateFn<External>(q);
  extUnnamed(q);
  inlineFn(q);
  funcObjects(q);
  staticNamed(q);
  sameNamedLocals(q);
  extNamed(q);
  explicitNames(q);
}
