// Packing SYCL host types must be diagnosed before they cross the runtime ABI.
// Packing confined to user types and device-only compilation remain valid.
// RUN: %clangxx %fsycl-host-only -target x86_64-unknown-linux-gnu -fsyntax-only -Xclang -verify=normal %s
// RUN: %clangxx %fsycl-host-only -target x86_64-unknown-linux-gnu -fsyntax-only -fpack-struct -Xclang -verify=packed -Xclang -verify-ignore-unexpected=note %s
// RUN: %clangxx %fsycl-host-only -target x86_64-unknown-linux-gnu -fsyntax-only -fpack-struct=2 -Xclang -verify=packed -Xclang -verify-ignore-unexpected=note %s
// RUN: %clangxx %fsycl-host-only -target x86_64-unknown-linux-gnu -fsyntax-only -fpack-struct=4 -Xclang -verify=packed -Xclang -verify-ignore-unexpected=note %s
// RUN: %clangxx %fsycl-host-only -target x86_64-unknown-linux-gnu -fsyntax-only -fpack-struct=8 -Xclang -verify=normal %s
// RUN: %clangxx %fsycl-host-only -target x86_64-unknown-linux-gnu -fsyntax-only -fpack-struct -fno-pack-struct -Xclang -verify=normal %s
// RUN: %clangxx %fsycl-host-only -target x86_64-unknown-linux-gnu -fsyntax-only -DPACK_BEFORE_COMMON_HEADER=1 -Xclang -verify=packed -Xclang -verify-ignore-unexpected=note %s
// RUN: %clangxx %fsycl-host-only -target x86_64-unknown-linux-gnu -fsyntax-only -DPACK_BEFORE_COMMON_HEADER=4 -Xclang -verify=packed -Xclang -verify-ignore-unexpected=note %s
// RUN: %clangxx %fsycl-host-only -target x86_64-unknown-linux-gnu -fsyntax-only -DPACK_BEFORE_COMMON_HEADER=8 -Xclang -verify=normal %s
// RUN: %clangxx %fsycl-host-only -target x86_64-unknown-linux-gnu -fsyntax-only -DPACK_USER_TYPE -Xclang -verify=normal %s
// RUN: %clangxx %fsycl-host-only -target i386-unknown-linux-gnu -fsyntax-only -fpack-struct=4 -Xclang -verify=normal %s
// RUN: %clangxx %fsycl-host-only -target i386-unknown-linux-gnu -fsyntax-only -fpack-struct=2 -Xclang -verify=packed -Xclang -verify-ignore-unexpected=note %s
// RUN: %clangxx --driver-mode=cl --target=x86_64-pc-windows-msvc /std:c++17 /Zs /Zp4 /D_DLL /I%sycl_include /clang:-Xclang /clang:-verify=packed /clang:-Xclang /clang:-verify-ignore-unexpected=note %s
// RUN: %clangxx --driver-mode=cl --target=x86_64-pc-windows-msvc /std:c++17 /Zs /Zp8 /D_DLL /I%sycl_include /clang:-Xclang /clang:-verify=normal %s
// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -fpack-struct -Xclang -verify=normal %s
// RUN: %clangxx -fsycl -fsyntax-only -DPACK_AFTER_COMMON_HEADER=1 -Xclang -verify=runtime -Xclang -verify-ignore-unexpected=note %s
// RUN: %clangxx -fsycl -fsyntax-only -DPACK_AFTER_COMMON_HEADER=4 -Xclang -verify=runtime -Xclang -verify-ignore-unexpected=note %s

// normal-no-diagnostics
// clang-format off
// packed-error@sycl/detail/defines_elementary.hpp:* {{Structure packing is incompatible with the supported SYCL runtime ABI. Remove -fpack-struct or -Zp, and do not include SYCL headers inside #pragma pack. Apply packing only to user-defined types instead.}}
#ifndef __SYCL_DEVICE_ONLY__
// runtime-error@sycl/detail/string_view.hpp:* {{Structure packing is incompatible with the supported SYCL runtime ABI.}}
// runtime-error@sycl/detail/compile_time_kernel_info.hpp:* {{Structure packing is incompatible with the supported SYCL runtime ABI.}}
// runtime-error@sycl/detail/kernel_launch_helper.hpp:* {{Structure packing is incompatible with the supported SYCL runtime ABI.}}
#else
// runtime-no-diagnostics
#endif
// clang-format on

#ifdef PACK_USER_TYPE
#pragma pack(push, 1)
struct UserType {
  char Byte;
  void *Pointer;
};
#pragma pack(pop)
#endif

#if PACK_BEFORE_COMMON_HEADER == 1
#pragma pack(push, 1)
#elif PACK_BEFORE_COMMON_HEADER == 4
#pragma pack(push, 4)
#elif PACK_BEFORE_COMMON_HEADER == 8
#pragma pack(push, 8)
#endif

#include <sycl/detail/defines_elementary.hpp>

#ifdef PACK_BEFORE_COMMON_HEADER
#pragma pack(pop)
#endif

#if PACK_AFTER_COMMON_HEADER == 1
#pragma pack(push, 1)
#elif PACK_AFTER_COMMON_HEADER == 4
#pragma pack(push, 4)
#endif

#ifdef PACK_AFTER_COMMON_HEADER
// The common header has already been included, possibly by an integration
// header.
#include <sycl/sycl.hpp>
#pragma pack(pop)
#endif

struct __attribute__((packed)) PackedUserType {
  char Byte;
  void *Pointer;
};
static_assert(sizeof(PackedUserType) == sizeof(char) + sizeof(void *));
