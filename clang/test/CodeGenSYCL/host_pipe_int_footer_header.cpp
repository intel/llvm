// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -fsycl-int-footer=%t.footer.h -fsycl-int-header=%t.header.h -fsycl-unique-prefix=THE_PREFIX %s -emit-llvm -o %t.ll
// RUN: FileCheck -input-file=%t.footer.h %s --check-prefix=CHECK-FOOTER
// RUN: FileCheck -input-file=%t.header.h %s --check-prefix=CHECK-HEADER
#include "sycl.hpp"

// Test cases below show that 'sycl-unique-id' LLVM IR attribute is attached to the
// global variable whose type is decorated with host_pipe attribute, and that a
// unique string is generated.

using namespace sycl::ext::intel::experimental;
using namespace sycl;
queue q;

void foo() {
  q.submit([&](handler &h) {
    h.single_task<class kernel_name_1>([=]() {
      host_pipe<class HPInt, int>::read();
      host_pipe<class HPFloat, int>::read();
    });
  });
}

// CHECK-HEADER: namespace sycl {
// CHECK-HEADER-NEXT: inline namespace _V1 {
// CHECK-HEADER-NEXT: namespace detail {
// CHECK-HEADER: namespace {
// CHECK-HEADER-NEXT: class __sycl_host_pipe_registration {
// CHECK-HEADER-NEXT: public:
// CHECK-HEADER-NEXT: __sycl_host_pipe_registration() noexcept;
// CHECK-HEADER-NEXT: };
// CHECK-HEADER-NEXT: __sycl_host_pipe_registration __sycl_host_pipe_registrar;
// CHECK-HEADER-NEXT: } // namespace
// CHECK-HEADER: } // namespace detail
// CHECK-HEADER: } // namespace _V1
// CHECK-HEADER: } // namespace sycl

// CHECK-FOOTER: #include <sycl/detail/defines_elementary.hpp>
// CHECK-FOOTER: #include <sycl/detail/host_pipe_map.hpp>
// CHECK-FOOTER-NEXT: namespace sycl::detail {
// CHECK-FOOTER: namespace {
// CHECK-FOOTER-NEXT: __sycl_host_pipe_registration::__sycl_host_pipe_registration() noexcept {

// CHECK-FOOTER: host_pipe_map::add((void *)&::sycl::ext::intel::experimental::host_pipe<HPInt, int>::__pipe, "THE_PREFIX____ZN4sycl3_V13ext5intel12experimental9host_pipeIZZZ3foovENKUlRNS0_7handlerEE_clES6_ENKUlvE_clEvE5HPIntiE6__pipeE");
// CHECK-FOOTER: host_pipe_map::add((void *)&::sycl::ext::intel::experimental::host_pipe<HPFloat, int>::__pipe, "THE_PREFIX____ZN4sycl3_V13ext5intel12experimental9host_pipeIZZZ3foovENKUlRNS0_7handlerEE_clES6_ENKUlvE_clEvE7HPFloatiE6__pipeE");

// CHECK-FOOTER: } // namespace (unnamed)
// CHECK-FOOTER: } // namespace sycl::detail

