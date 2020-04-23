// RUN: %clang_cc1 -fsycl -fsycl-is-device -fsycl-int-header=%t.h %s
// RUN: FileCheck -input-file=%t.h %s
//
// CHECK: #include <cstddef>
// CHECK-NEXT: #include <CL/sycl/detail/defines.hpp>
// CHECK-NEXT: #include <CL/sycl/detail/kernel_desc.hpp>
// CHECK-NEXT: using nullptr_t = std::nullptr_t;
//
// CHECK: static constexpr
// CHECK-NEXT: const char* const kernel_names[] = {
// CHECK-NEXT:   "_ZTSDn"
// CHECK-NEXT:   "_ZTSSt4byte"
// CHECK-NEXT:   "_ZTSm",
// CHECK-NEXT:   "_ZTSl"
// CHECK-NEXT: };
//
// CHECK: static constexpr
// CHECK-NEXT: const kernel_param_desc_t kernel_signatures[] = {
// CHECK-NEXT:   //--- _ZTSDn
// CHECK-EMPTY:
// CHECK-NEXT:   //--- _ZTSSt4byte
// CHECK-EMPTY:
// CHECK-NEXT:   //--- _ZTSm
// CHECK-EMPTY:
// CHECK-NEXT:   //--- _ZTSl
// CHECK-EMPTY:
// CHECK-NEXT: };
//
// CHECK: static constexpr
// CHECK-NEXT: const unsigned kernel_signature_start[] = {
// CHECK-NEXT:  0, // _ZTSDn
// CHECK-NEXT:  1, // _ZTSSt4byte
// CHECK-NEXT:  2, // _ZTSm
// CHECK-NEXT:  3 // _ZTSl
// CHECK-NEXT: };

// CHECK: template <> struct KernelInfo<nullptr_t> {
// CHECK: template <> struct KernelInfo<::std::byte> {
// CHECK: template <> struct KernelInfo<unsigned long> {
// CHECK: template <> struct KernelInfo<long> {

void usage() {
}

namespace std {
typedef long unsigned int size_t;
typedef long int ptrdiff_t;
typedef decltype(nullptr) nullptr_t;
enum class byte : unsigned char {};
} // namespace std

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<std::nullptr_t>([]() { usage(); });
  kernel_single_task<std::byte>([=]() {});
  kernel_single_task<std::size_t>([=]() {});
  kernel_single_task<std::ptrdiff_t>([=]() {});
  return 0;
}
