// clang-format off
// RUN: %clangxx -fsycl -c -emit-llvm -S -o - %s | FileCheck %s --check-prefix CHK-HOST
// RUN: %clangxx -fsycl -fsycl-device-only -O0 -c -emit-llvm -S -o - %s | FileCheck %s --check-prefix CHK-DEVICE
// REQUIRES: linux

#include <CL/sycl.hpp>

#ifdef __SYCL_DEVICE_ONLY__
// CHK-DEVICE: define dso_local spir_func void @_Z3accN2cl4sycl8accessorIiLi1ELNS0_6access4modeE1024ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE(%"class._ZTSN2cl4sycl8accessorIiLi1ELNS0_6access4modeE1024ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor"* byval(%"class._ZTSN2cl4sycl8accessorIiLi1ELNS0_6access4modeE1024ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor") align 8 %0)
SYCL_EXTERNAL void acc(sycl::accessor<int, 1, sycl::access::mode::read, sycl::access::target::global_buffer>) {} 

// CHK-DEVICE: define dso_local spir_func void @_Z3accN2cl4sycl8accessorIiLi1ELNS0_6access4modeE1024ELNS2_6targetE2016ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE(%"class._ZTSN2cl4sycl8accessorIiLi1ELNS0_6access4modeE1024ELNS2_6targetE2016ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor"* byval(%"class._ZTSN2cl4sycl8accessorIiLi1ELNS0_6access4modeE1024ELNS2_6targetE2016ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor") align 8 %0)
SYCL_EXTERNAL void acc(sycl::accessor<int, 1, sycl::access::mode::read, sycl::access::target::local>) {} 

// CHK-DEVICE: define dso_local spir_func void @_Z3accN2cl4sycl8accessorINS0_3vecIiLi4EEELi1ELNS0_6access4modeE1024ELNS4_6targetE2017ELNS4_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE(%"class._ZTSN2cl4sycl8accessorINS0_3vecIiLi4EEELi1ELNS0_6access4modeE1024ELNS4_6targetE2017ELNS4_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor"* byval(%"class._ZTSN2cl4sycl8accessorINS0_3vecIiLi4EEELi1ELNS0_6access4modeE1024ELNS4_6targetE2017ELNS4_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor") align 8 %0)
SYCL_EXTERNAL void acc(sycl::accessor<sycl::cl_int4, 1, sycl::access::mode::read, sycl::access::target::image>) {} 

// CHK-DEVICE: define dso_local spir_func void @_Z11private_memN2cl4sycl14private_memoryIiLi1EEE(%"class._ZTSN2cl4sycl14private_memoryIiLi1EEE.cl::sycl::private_memory"* byval(%"class._ZTSN2cl4sycl14private_memoryIiLi1EEE.cl::sycl::private_memory") align 4 %0)
SYCL_EXTERNAL void private_mem(sycl::private_memory<int, 1>) {};

// CHK-DEVICE: define dso_local spir_func void @_Z5rangeN2cl4sycl5rangeILi1EEE(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %0)
SYCL_EXTERNAL void range(sycl::range<1>) {}

// CHK-DEVICE: define dso_local spir_func void @_Z2idN2cl4sycl2idILi1EEE(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %0)
SYCL_EXTERNAL void id(sycl::id<1>) {}

// CHK-DEVICE: define dso_local spir_func void @_Z4itemN2cl4sycl2idILi1EEE(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %0)
SYCL_EXTERNAL void item(sycl::id<1>) {}

// CHK-DEVICE: define dso_local spir_func void @_Z3vecN2cl4sycl3vecIiLi16EEE(%"class._ZTSN2cl4sycl3vecIiLi16EEE.cl::sycl::vec"* byval(%"class._ZTSN2cl4sycl3vecIiLi16EEE.cl::sycl::vec") align 64 %0)
SYCL_EXTERNAL void vec(sycl::vec<int, 16>) {}

// CHK-DEVICE: define dso_local spir_func void @_Z6streamN2cl4sycl6streamE(%"class._ZTSN2cl4sycl6streamE.cl::sycl::stream"* byval(%"class._ZTSN2cl4sycl6streamE.cl::sycl::stream") align 8 %0)
SYCL_EXTERNAL void stream(sycl::stream) {}

// CHK-DEVICE: define dso_local spir_func void @_Z7samplerN2cl4sycl7samplerE(%"class._ZTSN2cl4sycl7samplerE.cl::sycl::sampler"* byval(%"class._ZTSN2cl4sycl7samplerE.cl::sycl::sampler") align 8 %0)
SYCL_EXTERNAL void sampler(sycl::sampler) {}
#else
// CHK-HOST: define dso_local void @_Z3accN2cl4sycl8accessorIiLi1ELNS0_6access4modeE1024ELNS2_6targetE2018ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE(%"class.cl::sycl::accessor"* %0)
void acc(sycl::accessor<int, 1, sycl::access::mode::read, sycl::access::target::host_buffer>) {} 

// CHK-HOST: define dso_local void @_Z3accN2cl4sycl8accessorIiLi1ELNS0_6access4modeE1024ELNS2_6targetE2016ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE(%"class.cl::sycl::accessor.3"* %0)
void acc(sycl::accessor<int, 1, sycl::access::mode::read, sycl::access::target::local>) {} 

// CHK-HOST: define dso_local void @_Z3accN2cl4sycl8accessorINS0_3vecIiLi4EEELi1ELNS0_6access4modeE1024ELNS4_6targetE2019ELNS4_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE(%"class.cl::sycl::accessor.8"* %0)
void acc(sycl::accessor<sycl::cl_int4, 1, sycl::access::mode::read, sycl::access::target::host_image>) {} 

// CHK-HOST: define dso_local void @_Z3bufN2cl4sycl6bufferIiLi1ENS0_6detail17aligned_allocatorIcEEvEE(%"class.cl::sycl::buffer"* %0)
void buf(sycl::buffer<int>) {}

// CHK-HOST: define dso_local void @_Z3ctxN2cl4sycl7contextE(%"class.cl::sycl::context"* %0)
void ctx(sycl::context) {}

// CHK-HOST: define dso_local void @_Z6deviceN2cl4sycl6deviceE(%"class.cl::sycl::device"* %0)
void device(sycl::device) {}

// CHK-HOST: define dso_local void @_Z10device_evtN2cl4sycl12device_eventE(i8** %.coerce)
void device_evt(sycl::device_event) {}

// CHK-HOST: define dso_local void @_Z5eventN2cl4sycl5eventE(%"class.cl::sycl::event"* %0)
void event(sycl::event) {}

// CHK-HOST: define dso_local void @_Z15device_selectorRN2cl4sycl15device_selectorE(%"class.cl::sycl::device_selector"* nonnull align 8 dereferenceable(8) %0)
void device_selector(sycl::device_selector&) {}

// CHK-HOST: define dso_local void @_Z7handlerRN2cl4sycl7handlerE(%"class.cl::sycl::handler"* nonnull align 8 dereferenceable(560) %0)
void handler(sycl::handler&) {}

// CHK-HOST: define dso_local void @_Z5imageN2cl4sycl5imageILi1ENS0_6detail17aligned_allocatorIhEEEE(%"class.cl::sycl::image"* %0)
void image(sycl::image<1>) {}

// CHK-HOST: define dso_local void @_Z5rangeN2cl4sycl5rangeILi1EEE(i64 %.coerce)
void range(sycl::range<1>) {}

// CHK-HOST: define dso_local void @_Z2idN2cl4sycl2idILi1EEE(i64 %.coerce)
void id(sycl::id<1>) {}

// CHK-HOST: define dso_local void @_Z4itemN2cl4sycl4itemILi1ELb1EEE(%"class.cl::sycl::item"* byval(%"class.cl::sycl::item") align 8 %0)
void item(sycl::item<1>) {}

// CHK-HOST: define dso_local void @_Z6streamN2cl4sycl6streamE(%"class.cl::sycl::stream"* %0)
void stream(sycl::stream) {}

// CHK-HOST: define dso_local void @_Z7samplerN2cl4sycl7samplerE(%"class.cl::sycl::sampler"* %0)
void sampler(sycl::sampler) {}

// CHK-HOST: define dso_local void @_Z5queueN2cl4sycl5queueE(%"class.cl::sycl::queue"* %0)
void queue(sycl::queue) {}

// CHK-HOST: define dso_local void @_Z7programN2cl4sycl7programE(%"class.cl::sycl::program"* %0)
void program(sycl::program) {}

// CHK-HOST: define dso_local void @_Z6kernelN2cl4sycl6kernelE(%"class.cl::sycl::kernel"* %0)
void kernel(sycl::kernel) {}

// CHK-HOST: define dso_local void @_Z8platformN2cl4sycl8platformE(%"class.cl::sycl::platform"* %0)
void platform(sycl::platform) {}

// CHK-HOST: define dso_local void @_Z3vecN2cl4sycl3vecIiLi16EEE(%"class.cl::sycl::vec"* %0)
void vec(sycl::vec<int, 16>) {}
#endif
