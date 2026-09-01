// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

// Pins which arguments the handler-less `nd_launch` overloads bind themselves
// and with which kind, since neither is observable from an end to end test.
// Every case here has to agree with `handler::setArgHelper`.

#include <sycl/sycl.hpp>

namespace oneapiext = sycl::ext::oneapi::experimental;
namespace ext_detail = sycl::ext::oneapi::experimental::detail;

using sycl::detail::kernel_param_kind_t;

enum class Sign : int { Plus = 1 };
struct Trivial {
  int A, B;
};

// Bound without a handler: no requirement for the scheduler to track.
static_assert(ext_detail::is_plain_kernel_arg_v<int>);
static_assert(ext_detail::is_plain_kernel_arg_v<const int &>);
static_assert(ext_detail::is_plain_kernel_arg_v<float>);
static_assert(ext_detail::is_plain_kernel_arg_v<Sign>);
static_assert(ext_detail::is_plain_kernel_arg_v<int *>);
static_assert(ext_detail::is_plain_kernel_arg_v<sycl::OpenCLMemT>);
static_assert(ext_detail::is_plain_kernel_arg_v<oneapiext::raw_kernel_arg>);
// An array of scalars is bound as the bytes it is, so it belongs on this path.
static_assert(ext_detail::is_plain_kernel_arg_v<int (&)[4]>);
static_assert(ext_detail::is_plain_kernel_arg_v<int (&)[2][3]>);

// Kept on the command group path.
static_assert(!ext_detail::is_plain_kernel_arg_v<sycl::accessor<int, 1>>);
static_assert(!ext_detail::is_plain_kernel_arg_v<sycl::local_accessor<int, 1>>);
// A class type may be a struct with special types inside, and an array of them
// would hide that just as well, so both keep using the handler.
static_assert(!ext_detail::is_plain_kernel_arg_v<Trivial>);
static_assert(!ext_detail::is_plain_kernel_arg_v<Trivial (&)[2]>);

// A pointer is an address, and has to be bound as one.
static_assert(ext_detail::plain_arg_kind_v<int *> ==
              kernel_param_kind_t::kind_pointer);
static_assert(ext_detail::plain_arg_kind_v<const void *> ==
              kernel_param_kind_t::kind_pointer);
// `cl_mem` is a pointer typedef but names a memory object, so it is bound as
// the bytes of the handle, the way the handler binds it.
static_assert(ext_detail::plain_arg_kind_v<sycl::OpenCLMemT> ==
              kernel_param_kind_t::kind_std_layout);
static_assert(ext_detail::plain_arg_kind_v<int> ==
              kernel_param_kind_t::kind_std_layout);
static_assert(ext_detail::plain_arg_kind_v<int[4]> ==
              kernel_param_kind_t::kind_std_layout);
static_assert(ext_detail::plain_arg_kind_v<oneapiext::raw_kernel_arg> ==
              kernel_param_kind_t::kind_std_layout);

int main() { return 0; }
