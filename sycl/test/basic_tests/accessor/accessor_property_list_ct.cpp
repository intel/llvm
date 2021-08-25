// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out

#include <CL/sycl.hpp>

#include <type_traits>

using namespace sycl::ext::oneapi;

void foo(sycl::accessor<int, 1, sycl::access::mode::read_write,
                        sycl::access::target::global_buffer,
                        sycl::access::placeholder::true_t,
                        accessor_property_list<property::no_alias::instance<>,
                                               property::no_offset::instance<>>>
             acc) {}

int main() {
  {
    // Create empty property list
    accessor_property_list PL;
  }

  {
    // No properties
    accessor_property_list PL{};
    static_assert(!PL.has_property<property::no_alias>(), "Property is found");
    static_assert(!PL.has_property<property::no_offset>(), "Property is found");
  }

  {
    // Single CT property
    accessor_property_list PL{no_alias};
    static_assert(PL.has_property<property::no_alias>(), "Property not found");
    static_assert(!PL.has_property<property::no_offset>(), "Property is found");
  }

  {
    // Multiple CT properties
    accessor_property_list PL{no_alias, no_offset};
    static_assert(PL.has_property<property::no_alias>(), "Property not found");
    static_assert(PL.has_property<property::no_offset>(), "Property not found");

    static_assert(PL.get_property<property::no_alias>() == no_alias,
                  "Properties are not equal");
  }

  {
    // Property list copy
    accessor_property_list PL{no_alias, sycl::no_init};

    accessor_property_list PL_1{PL};
    static_assert(PL_1.has_property<property::no_alias>(),
                  "Property not found");
  }

  {
    // Conversion
    accessor_property_list PL{no_offset, no_alias};
    int *data = nullptr;
    sycl::buffer<int, 1> buf_data(data, sycl::range<1>(1),
                                  {sycl::property::buffer::use_host_ptr()});

    sycl::accessor acc_1(buf_data, PL);
    foo(acc_1);
  }

  {
    int data[1] = {0};

    sycl::buffer<int, 1> buf_data(data, sycl::range<1>(1),
                                  {sycl::property::buffer::use_host_ptr()});
    sycl::queue Queue;

    Queue.submit([&](sycl::handler &cgh) {
      accessor_property_list PL{no_alias};
      sycl::accessor acc_1(buf_data, cgh, PL);
      sycl::accessor acc_2(buf_data, cgh, sycl::range<1>(0), PL);
      sycl::accessor acc_3(buf_data, cgh, sycl::range<1>(0), sycl::id<1>(1),
                           PL);
      sycl::accessor acc_4(buf_data, cgh, sycl::read_only, PL);
      sycl::accessor acc_5(buf_data, cgh, sycl::range<1>(0), sycl::read_only,
                           PL);
      sycl::accessor acc_6(buf_data, cgh, sycl::range<1>(0), sycl::id<1>(0),
                           sycl::read_only, PL);
      sycl::accessor acc_7(buf_data, cgh, sycl::write_only, PL);
      sycl::accessor acc_8(buf_data, cgh, sycl::range<1>(0), sycl::write_only,
                           PL);
      sycl::accessor acc_9(buf_data, cgh, sycl::range<1>(0), sycl::id<1>(0),
                           sycl::write_only, PL);
      cgh.single_task<class NullKernel>([]() {});
    });
    accessor_property_list PL{no_alias};
    sycl::accessor acc_1(buf_data, PL);
    sycl::accessor acc_2(buf_data, sycl::range<1>(0), PL);
    sycl::accessor acc_3(buf_data, sycl::range<1>(0), sycl::id<1>(0), PL);
    sycl::accessor acc_4(buf_data, sycl::read_only, PL);
    sycl::accessor acc_5(buf_data, sycl::range<1>(0), sycl::read_only, PL);
    sycl::accessor acc_6(buf_data, sycl::range<1>(0), sycl::id<1>(0),
                         sycl::read_only, PL);
    sycl::accessor acc_7(buf_data, sycl::write_only, PL);
    sycl::accessor acc_8(buf_data, sycl::range<1>(0), sycl::write_only, PL);
    sycl::accessor acc_9(buf_data, sycl::range<1>(0), sycl::id<1>(0),
                         sycl::write_only, PL);
  }

  {
    int data[1] = {0};

    sycl::buffer<int, 1> buf_data(data, sycl::range<1>(1),
                                  {sycl::property::buffer::use_host_ptr()});

    accessor_property_list PL{sycl::no_init, no_alias};
    sycl::accessor acc_1(buf_data, PL);
    sycl::accessor<int, 1, sycl::access::mode::read_write,
                   sycl::access::target::global_buffer,
                   sycl::access::placeholder::true_t,
                   accessor_property_list<property::no_alias::instance<>>>
        acc_2(acc_1);
  }

  return 0;
}
