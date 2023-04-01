// RUN: %clang_cc1 -fsycl-is-device -verify %s

// Diagnostic tests for sycl_type() attribute

// expected-error@+1{{'sycl_type' attribute only applies to classes}}
[[__sycl_detail__::sycl_type(accessor)]] int a;

// expected-error@+1{{'sycl_type' attribute only applies to classes}}
[[__sycl_detail__::sycl_type(accessor)]] void func1() {}

// expected-error@+1{{'sycl_type' attribute requires an identifier}}
class [[__sycl_detail__::sycl_type("accessor")]] A {};

// expected-error@+1{{'sycl_type' attribute takes one argument}}
class [[__sycl_detail__::sycl_type()]] B {};

// expected-error@+1{{'sycl_type' attribute argument 'NotValidType' is not supported}}
class [[__sycl_detail__::sycl_type(NotValidType)]] C {};

// expected-note@+1{{previous attribute is here}}
class [[__sycl_detail__::sycl_type(spec_constant)]] spec_constant;
// expected-error@+1{{attribute 'sycl_type' is already applied with different arguments}}
class [[__sycl_detail__::sycl_type(accessor)]] spec_constant {};

// expected-error@+2{{attribute 'sycl_type' is already applied with different arguments}}
// expected-note@+1{{previous attribute is here}}
class [[__sycl_detail__::sycl_type(group)]] [[__sycl_detail__::sycl_type(accessor)]] group {};

// Valid usage -

class [[__sycl_detail__::sycl_type(accessor)]] accessor {};

template <typename T>
class [[__sycl_detail__::sycl_type(local_accessor)]] local_accessor {};

enum class [[__sycl_detail__::sycl_type(aspect)]] aspect {};

// No diagnostic for matching arguments. 
class [[__sycl_detail__::sycl_type(kernel_handler)]] kernel_handler;
class [[__sycl_detail__::sycl_type(kernel_handler)]]
[[__sycl_detail__::sycl_type(kernel_handler)]] kernel_handler {};
class kernel_handler;




