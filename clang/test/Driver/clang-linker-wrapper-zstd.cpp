// REQUIRES: zstd && system-linux && x86-registered-target

// clang-linker-wrapper compression test: checks that the wrapper compresses
// SYCL device images when --compress is set, tags them with
// SYCLBinaryImageFormat::BIF_Compressed (int8 value 4) in the emitted wrapper
// module, honors --compression-level=, and reports invalid values.

// Prepare test data. The bitcode is over the 512-byte compression
// threshold, so the compression actually runs.
// RUN: %clang -cc1 -fsycl-is-device -disable-llvm-passes -triple=spir64-unknown-unknown %s -emit-llvm-bc -o %t.device.bc
// RUN: llvm-offload-binary -o %t.fat --image=file=%t.device.bc,kind=sycl,triple=spir64-unknown-unknown
// RUN: %clang -cc1 %s -triple=x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.fat
// RUN: touch %t.devicelib.bc

// With --compress and --wrapper-verbose the compression path fires and the
// wrapped image's Format slot in %__sycl.tgt_device_image is i8 4
// (BIF_Compressed).
// RUN: clang-linker-wrapper --print-wrapped-module --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --bitcode-library=spir64-unknown-unknown=%t.devicelib.bc \
// RUN:   -sycl-post-link-options="-split=auto -symbols -properties" \
// RUN:   --compress --compression-level=9 --wrapper-verbose \
// RUN:   %t.o -o %t.out --linker-path="/usr/bin/ld" 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK-COMPRESS

// Capture the original and compressed sizes.
// CHECK-COMPRESS: [Compression] Original image size: [[#ORIG:]]
// CHECK-COMPRESS: [Compression] Compressed image size: [[#COMP:]]
// CHECK-COMPRESS: [Compression] Compression level used: 9
//
// Assert that the compressed size is strictly smaller than the original size
// (i.e. ORIG - COMP > 0). FileCheck's numeric matching form only supports the
// `==` constraint — there is no `<` or `>` — but sub() rejects underflow at
// expression-evaluation time. We attach sub(ORIG, COMP) to a CHECK-NOT
// pattern that begins with a sentinel string never present in the output:
//   * when ORIG > COMP, sub() succeeds; the substituted pattern is
//     "COMPRESSION_SIZE_CHECK<some-number>", which doesn't appear, so the
//     CHECK-NOT is satisfied vacuously.
//   * when COMP >= ORIG, sub() underflows and FileCheck fails the pattern.
// CHECK-COMPRESS-NOT: COMPRESSION_SIZE_CHECK[[#sub(ORIG, COMP)]]
// The tgt_device_image struct is { i16 Version, i8 Kind, i8 Format, ptr ... };
// Kind for SYCL is 4, Format for BIF_Compressed is also 4.
// CHECK-COMPRESS: @.sycl_offloading.device_images = internal unnamed_addr constant [1 x %__sycl.tgt_device_image] [%__sycl.tgt_device_image { i16 {{[0-9]+}}, i8 4, i8 4,

// Without --compress the image is left untagged (Format = BIF_None = 0) and
// no [Compression] verbose lines are emitted.
// RUN: clang-linker-wrapper --print-wrapped-module --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --bitcode-library=spir64-unknown-unknown=%t.devicelib.bc \
// RUN:   -sycl-post-link-options="-split=auto -symbols -properties" \
// RUN:   --wrapper-verbose \
// RUN:   %t.o -o %t.out --linker-path="/usr/bin/ld" 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK-NO-COMPRESS

// CHECK-NO-COMPRESS-NOT: [Compression]
// CHECK-NO-COMPRESS: @.sycl_offloading.device_images = internal unnamed_addr constant [1 x %__sycl.tgt_device_image] [%__sycl.tgt_device_image { i16 {{[0-9]+}}, i8 4, i8 0,

// A non-integer --compression-level= is diagnosed.
// RUN: not clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --bitcode-library=spir64-unknown-unknown=%t.devicelib.bc \
// RUN:   -sycl-post-link-options="-split=auto -symbols -properties" \
// RUN:   --compress --compression-level=notanumber \
// RUN:   %t.o -o %t.out --linker-path="/usr/bin/ld" 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK-BAD-LEVEL

// CHECK-BAD-LEVEL: invalid value for --offload-compression-level=: 'notanumber'

template <typename t, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &func) {
    func();
}

extern "C" {
// Symbols so the linker doesn't fail resolving them.
void __sycl_register_lib(void *) {}
void __sycl_unregister_lib(void *) {}
}

int main() {
    kernel<class fake_kernel>([](){});
}
