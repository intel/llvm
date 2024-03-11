// RUN: %clangxx -fsycl -fsyntax-only -fno-color-diagnostics -Xclang -fdump-record-layouts %s | FileCheck %s
// REQUIRES: linux
// UNSUPPORTED: libcxx

// clang-format off

#include <sycl/image.hpp>

sycl::image<2> Img1{sycl::image_channel_order::rgba, sycl::image_channel_type::fp16, sycl::range<2>{10, 10}};

// CHECK: 0 | class sycl::image<2>
// CHECK-NEXT: 0 |   class sycl::detail::unsampled_image_common<2, class sycl::detail::aligned_allocator<unsigned char> > (base)
// CHECK-NEXT: 0 |     class sycl::detail::image_common<2, class sycl::detail::aligned_allocator<unsigned char> > (base)
// CHECK-NEXT: 0 |       class sycl::detail::image_plain (base)
// CHECK-NEXT: 0 |         class std::shared_ptr<class sycl::detail::image_impl> impl
// CHECK-NEXT: 0 |           class std::__shared_ptr<class sycl::detail::image_impl> (base)
// CHECK-NEXT: 0 |             class std::__shared_ptr_access<class sycl::detail::image_impl, __gnu_cxx::_S_atomic> (base) (empty)
// CHECK-NEXT: 0 |             element_type * _M_ptr
// CHECK-NEXT: 8 |             class std::__shared_count<> _M_refcount
// CHECK-NEXT: 8 |               _Sp_counted_base<(_Lock_policy)2U> * _M_pi
// CHECK-NEXT:   | [sizeof=16, dsize=16, align=8,
// CHECK-NEXT:   |  nvsize=16, nvalign=8]

sycl::unsampled_image<2> Img2{sycl::image_format::r16b16g16a16_sfloat, sycl::range<2>{10, 10}};

// CHECK: 0 | class sycl::unsampled_image<2>
// CHECK-NEXT: 0 |   class sycl::detail::unsampled_image_common<2, class sycl::detail::aligned_allocator<unsigned char> > (base)
// CHECK-NEXT: 0 |     class sycl::detail::image_common<2, class sycl::detail::aligned_allocator<unsigned char> > (base)
// CHECK-NEXT: 0 |       class sycl::detail::image_plain (base)
// CHECK-NEXT: 0 |         class std::shared_ptr<class sycl::detail::image_impl> impl
// CHECK-NEXT: 0 |           class std::__shared_ptr<class sycl::detail::image_impl> (base)
// CHECK-NEXT: 0 |             class std::__shared_ptr_access<class sycl::detail::image_impl, __gnu_cxx::_S_atomic> (base) (empty)
// CHECK-NEXT: 0 |             element_type * _M_ptr
// CHECK-NEXT: 8 |             class std::__shared_count<> _M_refcount
// CHECK-NEXT: 8 |               _Sp_counted_base<(_Lock_policy)2U> * _M_pi
// CHECK-NEXT: 0 |   class sycl::detail::OwnerLessBase<class sycl::unsampled_image<2> > (base) (empty)
// CHECK-NEXT:   | [sizeof=16, dsize=16, align=8,
// CHECK-NEXT:   |  nvsize=16, nvalign=8]

sycl::half Data[10*10*4];
sycl::image_sampler Sampler{
  sycl::addressing_mode::none,
  sycl::coordinate_normalization_mode::unnormalized,
  sycl::filtering_mode::linear};
sycl::sampled_image<2> Img3{Data, sycl::image_format::r16b16g16a16_sfloat, Sampler, sycl::range<2>{10, 10}};


// CHECK: 0 | class sycl::sampled_image<2>
// CHECK-NEXT: 0 |   class sycl::detail::image_common<2, class sycl::detail::aligned_allocator<unsigned char> > (base)
// CHECK-NEXT: 0 |     class sycl::detail::image_plain (base)
// CHECK-NEXT: 0 |       class std::shared_ptr<class sycl::detail::image_impl> impl
// CHECK-NEXT: 0 |         class std::__shared_ptr<class sycl::detail::image_impl> (base)
// CHECK-NEXT: 0 |           class std::__shared_ptr_access<class sycl::detail::image_impl, __gnu_cxx::_S_atomic> (base) (empty)
// CHECK-NEXT: 0 |           element_type * _M_ptr
// CHECK-NEXT: 8 |           class std::__shared_count<> _M_refcount
// CHECK-NEXT: 8 |             _Sp_counted_base<(_Lock_policy)2U> * _M_pi
// CHECK-NEXT: 0 |   class sycl::detail::OwnerLessBase<class sycl::sampled_image<2> > (base) (empty)
// CHECK-NEXT:   | [sizeof=16, dsize=16, align=8,
// CHECK-NEXT:   |  nvsize=16, nvalign=8]
