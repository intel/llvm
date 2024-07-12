// RUN: %clangxx -fsycl -c -fno-color-diagnostics -Xclang -fdump-record-layouts %s -o %t.out | FileCheck %s
// REQUIRES: linux
// UNSUPPORTED: libcxx

#include <sycl/exception.hpp>
#include <sycl/queue.hpp>

void foo() {
  try {
    sycl::queue q;
  } catch (sycl::exception &e) {
  }
}

// The order of field declarations and their types are important.
// CHECK-LABEL:        0 | class sycl::exception
// CHECK-NEXT:         8 |   class std::shared_ptr<class sycl::detail::string> MMsg
// CHECK-NEXT:         8 |     class std::__shared_ptr<class sycl::detail::string> (base)
// CHECK-NEXT:         8 |       class std::__shared_ptr_access<class sycl::detail::string, __gnu_cxx::_S_atomic> (base) (empty)
// CHECK-NEXT:         8 |       element_type * _M_ptr
// CHECK-NEXT:        16 |       class std::__shared_count<> _M_refcount
// CHECK-NEXT:        16 |         _Sp_counted_base<(_Lock_policy)2U> * _M_pi
// CHECK-NEXT:        24 |   pi_int32 MPIErr
// CHECK-NEXT:        32 |   class std::shared_ptr<class sycl::context> MContext
// CHECK-NEXT:        32 |     class std::__shared_ptr<class sycl::context> (base)
// CHECK-NEXT:        32 |       class std::__shared_ptr_access<class sycl::context, __gnu_cxx::_S_atomic> (base) (empty)
// CHECK-NEXT:        32 |       element_type * _M_ptr
// CHECK-NEXT:        40 |       class std::__shared_count<> _M_refcount
// CHECK-NEXT:        40 |         _Sp_counted_base<(_Lock_policy)2U> * _M_pi
// CHECK-NEXT:        48 |   {{class|struct}} std::error_code MErrC
// CHECK-NEXT:        48 |     int _M_value
// CHECK-NEXT:        56 |     const error_category * _M_cat
// CHECK-NEXT:         0 |   class std::exception (primary virtual base)
// CHECK-NEXT:         0 |     (exception vtable pointer)
// CHECK-NEXT:           | [sizeof=64, dsize=64, align=8,
