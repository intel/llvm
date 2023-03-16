// RUN: %clangxx -fsycl -fsyntax-only %s
//
// Tests that casting multi_ptr to and from generic compiles for various
// combinations of valid qualifiers.

#include <sycl/sycl.hpp>

using namespace sycl;

template <typename T, access::address_space AddrSpace,
          sycl::access::decorated IsDecorated>
void test(queue &Q) {
  T *GlobPtr = malloc_device<T>(1, Q);
  Q.submit([&](handler &CGH) {
     local_accessor<T> LocPtr{1, CGH};
     CGH.single_task([=]() {
       T X = 0;
       T *InPtr;
       if constexpr (AddrSpace == access::address_space::global_space)
         InPtr = GlobPtr;
       else if constexpr (AddrSpace == access::address_space::local_space)
         InPtr = LocPtr.get_pointer();
       else
         InPtr = &X;

       auto MPtr = address_space_cast<AddrSpace, IsDecorated>(InPtr);
       multi_ptr<T, access::address_space::generic_space, IsDecorated> GenPtr;
       GenPtr = MPtr;
       MPtr = multi_ptr<T, AddrSpace, IsDecorated>{GenPtr};
     });
   }).wait();
}

template <typename T, access::address_space AddrSpace>
void testAllDecos(queue &Q) {
  test<T, AddrSpace, sycl::access::decorated::yes>(Q);
  test<T, AddrSpace, sycl::access::decorated::no>(Q);
}

template <typename T> void testAllAddrSpace(queue &Q) {
  testAllDecos<T, access::address_space::private_space>(Q);
  testAllDecos<T, access::address_space::local_space>(Q);
  testAllDecos<T, access::address_space::global_space>(Q);
}

template <typename T> void testAllQuals(queue &Q) {
  using UnqualT = std::remove_cv_t<T>;
  testAllAddrSpace<UnqualT>(Q);
  testAllAddrSpace<std::add_const_t<UnqualT>>(Q);
  testAllAddrSpace<std::add_volatile_t<UnqualT>>(Q);
  testAllAddrSpace<std::add_cv_t<UnqualT>>(Q);
}

int main() {
  queue Q;
  testAllQuals<bool>(Q);
  testAllQuals<char>(Q);
  testAllQuals<short>(Q);
  testAllQuals<int>(Q);
  testAllQuals<long>(Q);
  testAllQuals<long long>(Q);
  testAllQuals<sycl::half>(Q);
  testAllQuals<float>(Q);
  testAllQuals<double>(Q);
  return 0;
}
