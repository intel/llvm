// RUN: %clangxx -std=c++17 -I %sycl_include -I %sycl_include/sycl -fsycl-device-only -c -fno-color-diagnostics -Xclang -fdump-record-layouts %s -o %t.out | grep -Pzo "0 \| class sycl::.*\n([^\n].*\n)*" | sort -z | FileCheck --implicit-check-not "{{std::basic_string|std::list}}" %s
// RUN: %clangxx -std=c++17 -I %sycl_include -I %sycl_include/sycl -c -fno-color-diagnostics -Xclang -fdump-record-layouts %s -o %t.out | grep -Pzo "0 \| class sycl::.*\n([^\n].*\n)*" | sort -z | FileCheck %s
// REQUIRES: linux
// UNSUPPORTED: libcxx

// The purpose of this test is to check that classes in sycl namespace that are
// defined in SYCL headers don't have std::string and std::list data members to
// avoid having the dual ABI issue (see
// https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html). I.e. if
// application is built with the old ABI and such data member is crossing ABI
// boundary then it will result in issues as SYCL RT is using new ABI by
// default. All such data members can potentially cross ABI boundaries and
// that's why we need to be sure that we use only ABI-neutral data members.

// New exclusions are NOT ALLOWED to this file unless it is guaranteed that data
// member is not crossing ABI boundary. All current exclusions are listed below.



// CHECK: 0 | class sycl::detail::CG
// CHECK-NEXT:          0 |   (CG vtable pointer)
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       136 |   class std::basic_string<char> MFunctionName
// CHECK-NEXT:       136 |     struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       152 |       union std::basic_string<char>::(anonymous at
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       168 |   class std::basic_string<char> MFileName
// CHECK-NEXT:       168 |     struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       184 |     union std::basic_string<char>::(anonymous at

// CHECK: 0 | class sycl::detail::CGAdviseUSM
// CHECK-NEXT:         0 |   class sycl::detail::CG (primary base)
// CHECK-NEXT:         0 |     (CG vtable pointer)
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       136 |     class std::basic_string<char> MFunctionName
// CHECK-NEXT:       136 |       struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       152 |       union std::basic_string<char>::(anonymous at
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       168 |     class std::basic_string<char> MFileName
// CHECK-NEXT:       168 |       struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       184 |       union std::basic_string<char>::(anonymous at

// CHECK: 0 | class sycl::detail::CGBarrier
// CHECK-NEXT:         0 |   class sycl::detail::CG (primary base)
// CHECK-NEXT:         0 |     (CG vtable pointer)
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       136 |     class std::basic_string<char> MFunctionName
// CHECK-NEXT:       136 |       struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       152 |       union std::basic_string<char>::(anonymous at
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       168 |     class std::basic_string<char> MFileName
// CHECK-NEXT:       168 |       struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       184 |       union std::basic_string<char>::(anonymous at

// CHECK: 0 | class sycl::detail::CGCopy
// CHECK-NEXT:         0 |   class sycl::detail::CG (primary base)
// CHECK-NEXT:         0 |     (CG vtable pointer)
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       136 |     class std::basic_string<char> MFunctionName
// CHECK-NEXT:       136 |       struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       152 |       union std::basic_string<char>::(anonymous at
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       168 |     class std::basic_string<char> MFileName
// CHECK-NEXT:       168 |       struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       184 |       union std::basic_string<char>::(anonymous at

// CHECK: 0 | class sycl::detail::CGCopy2DUSM
// CHECK-NEXT:         0 |   class sycl::detail::CG (primary base)
// CHECK-NEXT:         0 |     (CG vtable pointer)
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       136 |     class std::basic_string<char> MFunctionName
// CHECK-NEXT:       136 |       struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       152 |       union std::basic_string<char>::(anonymous at
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       168 |     class std::basic_string<char> MFileName
// CHECK-NEXT:       168 |       struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       184 |       union std::basic_string<char>::(anonymous at

// CHECK: 0 | class sycl::detail::CGCopyFromDeviceGlobal
// CHECK-NEXT:         0 |   class sycl::detail::CG (primary base)
// CHECK-NEXT:         0 |     (CG vtable pointer)
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       136 |     class std::basic_string<char> MFunctionName
// CHECK-NEXT:       136 |       struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       152 |       union std::basic_string<char>::(anonymous at
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       168 |     class std::basic_string<char> MFileName
// CHECK-NEXT:       168 |       struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       184 |       union std::basic_string<char>::(anonymous at

// CHECK: 0 | class sycl::detail::CGCopyImage
// CHECK-NEXT:         0 |   class sycl::detail::CG (primary base)
// CHECK-NEXT:         0 |     (CG vtable pointer)
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       136 |     class std::basic_string<char> MFunctionName
// CHECK-NEXT:       136 |       struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       152 |       union std::basic_string<char>::(anonymous at
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       168 |     class std::basic_string<char> MFileName
// CHECK-NEXT:       168 |       struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       184 |       union std::basic_string<char>::(anonymous at

// CHECK: 0 | class sycl::detail::CGCopyToDeviceGlobal
// CHECK-NEXT:         0 |   class sycl::detail::CG (primary base)
// CHECK-NEXT:         0 |     (CG vtable pointer)
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       136 |     class std::basic_string<char> MFunctionName
// CHECK-NEXT:       136 |       struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       152 |       union std::basic_string<char>::(anonymous at
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       168 |     class std::basic_string<char> MFileName
// CHECK-NEXT:       168 |       struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       184 |       union std::basic_string<char>::(anonymous at

// CHECK: 0 | class sycl::detail::CGCopyUSM
// CHECK-NEXT:         0 |   class sycl::detail::CG (primary base)
// CHECK-NEXT:         0 |     (CG vtable pointer)
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       136 |     class std::basic_string<char> MFunctionName
// CHECK-NEXT:       136 |       struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       152 |       union std::basic_string<char>::(anonymous at
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       168 |     class std::basic_string<char> MFileName
// CHECK-NEXT:       168 |       struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       184 |       union std::basic_string<char>::(anonymous at

// CHECK: 0 | class sycl::detail::CGExecCommandBuffer
// CHECK-NEXT:         0 |   class sycl::detail::CG (primary base)
// CHECK-NEXT:         0 |     (CG vtable pointer)
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       136 |     class std::basic_string<char> MFunctionName
// CHECK-NEXT:       136 |       struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       152 |       union std::basic_string<char>::(anonymous at
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       168 |     class std::basic_string<char> MFileName
// CHECK-NEXT:       168 |       struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       184 |       union std::basic_string<char>::(anonymous at

// CHECK: 0 | class sycl::detail::CGExecKernel
// CHECK-NEXT:         0 |   class sycl::detail::CG (primary base)
// CHECK-NEXT:         0 |     (CG vtable pointer)
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       136 |     class std::basic_string<char> MFunctionName
// CHECK-NEXT:       136 |       struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       152 |       union std::basic_string<char>::(anonymous at
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       168 |     class std::basic_string<char> MFileName
// CHECK-NEXT:       168 |       struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       184 |       union std::basic_string<char>::(anonymous at
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       408 |   class std::basic_string<char> MKernelName
// CHECK-NEXT:       408 |     struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       424 |     union std::basic_string<char>::(anonymous at

// CHECK: 0 | class sycl::detail::CGFill
// CHECK-NEXT:         0 |   class sycl::detail::CG (primary base)
// CHECK-NEXT:         0 |     (CG vtable pointer)
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       136 |     class std::basic_string<char> MFunctionName
// CHECK-NEXT:       136 |       struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       152 |       union std::basic_string<char>::(anonymous at
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       168 |     class std::basic_string<char> MFileName
// CHECK-NEXT:       168 |       struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       184 |       union std::basic_string<char>::(anonymous at

// CHECK: 0 | class sycl::detail::CGFill2DUSM
// CHECK-NEXT:         0 |   class sycl::detail::CG (primary base)
// CHECK-NEXT:         0 |     (CG vtable pointer)
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       136 |     class std::basic_string<char> MFunctionName
// CHECK-NEXT:       136 |       struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       152 |       union std::basic_string<char>::(anonymous at
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       168 |     class std::basic_string<char> MFileName
// CHECK-NEXT:       168 |       struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       184 |       union std::basic_string<char>::(anonymous at

// CHECK: 0 | class sycl::detail::CGFillUSM
// CHECK-NEXT:         0 |   class sycl::detail::CG (primary base)
// CHECK-NEXT:         0 |     (CG vtable pointer)
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       136 |     class std::basic_string<char> MFunctionName
// CHECK-NEXT:       136 |       struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       152 |       union std::basic_string<char>::(anonymous at
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       168 |     class std::basic_string<char> MFileName
// CHECK-NEXT:       168 |       struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       184 |       union std::basic_string<char>::(anonymous at

// CHECK: 0 | class sycl::detail::CGHostTask
// CHECK-NEXT:         0 |   class sycl::detail::CG (primary base)
// CHECK-NEXT:         0 |     (CG vtable pointer)
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       136 |     class std::basic_string<char> MFunctionName
// CHECK-NEXT:       136 |       struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       152 |       union std::basic_string<char>::(anonymous at
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       168 |     class std::basic_string<char> MFileName
// CHECK-NEXT:       168 |       struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       184 |       union std::basic_string<char>::(anonymous at

// CHECK: 0 | class sycl::detail::CGMemset2DUSM
// CHECK-NEXT:         0 |   class sycl::detail::CG (primary base)
// CHECK-NEXT:         0 |     (CG vtable pointer)
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       136 |     class std::basic_string<char> MFunctionName
// CHECK-NEXT:       136 |       struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       152 |       union std::basic_string<char>::(anonymous at
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       168 |     class std::basic_string<char> MFileName
// CHECK-NEXT:       168 |       struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       184 |       union std::basic_string<char>::(anonymous at

// CHECK: 0 | class sycl::detail::CGPrefetchUSM
// CHECK-NEXT:         0 |   class sycl::detail::CG (primary base)
// CHECK-NEXT:         0 |     (CG vtable pointer)
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       136 |     class std::basic_string<char> MFunctionName
// CHECK-NEXT:       136 |       struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       152 |       union std::basic_string<char>::(anonymous at
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       168 |     class std::basic_string<char> MFileName
// CHECK-NEXT:       168 |       struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       184 |       union std::basic_string<char>::(anonymous at

// CHECK: 0 | class sycl::detail::CGProfilingTag
// CHECK-NEXT:         0 |   class sycl::detail::CG (primary base)
// CHECK-NEXT:         0 |     (CG vtable pointer)
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       136 |     class std::basic_string<char> MFunctionName
// CHECK-NEXT:       136 |       struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       152 |       union std::basic_string<char>::(anonymous at
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       168 |     class std::basic_string<char> MFileName
// CHECK-NEXT:       168 |       struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       184 |       union std::basic_string<char>::(anonymous at

// CHECK: 0 | class sycl::detail::CGReadWriteHostPipe
// CHECK-NEXT:         0 |   class sycl::detail::CG (primary base)
// CHECK-NEXT:         0 |     (CG vtable pointer)
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       136 |     class std::basic_string<char> MFunctionName
// CHECK-NEXT:       136 |       struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       152 |       union std::basic_string<char>::(anonymous at
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       168 |     class std::basic_string<char> MFileName
// CHECK-NEXT:       168 |       struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       184 |       union std::basic_string<char>::(anonymous at
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       208 |   class std::basic_string<char> PipeName
// CHECK-NEXT:       208 |     struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       224 |       union std::basic_string<char>::(anonymous at

// CHECK: 0 | class sycl::detail::CGSemaphoreSignal
// CHECK-NEXT:         0 |   class sycl::detail::CG (primary base)
// CHECK-NEXT:         0 |     (CG vtable pointer)
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       136 |     class std::basic_string<char> MFunctionName
// CHECK-NEXT:       136 |       struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       152 |       union std::basic_string<char>::(anonymous at
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       168 |     class std::basic_string<char> MFileName
// CHECK-NEXT:       168 |       struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       184 |       union std::basic_string<char>::(anonymous at

// CHECK: 0 | class sycl::detail::CGSemaphoreWait
// CHECK-NEXT:         0 |   class sycl::detail::CG (primary base)
// CHECK-NEXT:         0 |     (CG vtable pointer)
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       136 |     class std::basic_string<char> MFunctionName
// CHECK-NEXT:       136 |       struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       152 |       union std::basic_string<char>::(anonymous at
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       168 |     class std::basic_string<char> MFileName
// CHECK-NEXT:       168 |       struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       184 |       union std::basic_string<char>::(anonymous at

// CHECK: 0 | class sycl::detail::CGUpdateHost
// CHECK-NEXT:         0 |   class sycl::detail::CG (primary base)
// CHECK-NEXT:         0 |     (CG vtable pointer)
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       136 |     class std::basic_string<char> MFunctionName
// CHECK-NEXT:       136 |       struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       152 |       union std::basic_string<char>::(anonymous at
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       168 |     class std::basic_string<char> MFileName
// CHECK-NEXT:       168 |       struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NOT:  {{^0 \| class|std::basic_string|std::list}}
// CHECK:       184 |       union std::basic_string<char>::(anonymous at

#include <sycl/sycl.hpp>
