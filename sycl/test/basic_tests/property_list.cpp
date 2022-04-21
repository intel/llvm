// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %RUN_ON_HOST %t.out
//
// CHECK: PASSED

// This test performs basic check of the SYCL property_list class.

#include <CL/sycl.hpp>
#include <iostream>

namespace sycl_property = cl::sycl::property;

int main() {
  bool Failed = false;

  {
    cl::sycl::property_list Empty{};
    if (Empty.has_property<sycl_property::buffer::use_host_ptr>()) {
      std::cerr << "Error: empty property list has property." << std::endl;
      Failed = true;
    }
  }

  {
    cl::sycl::context SYCLContext;
    sycl_property::buffer::context_bound ContextBound(SYCLContext);

    cl::sycl::property_list SeveralProps{sycl_property::image::use_host_ptr(),
                                         sycl_property::buffer::use_host_ptr(),
                                         sycl_property::image::use_host_ptr(),
                                         ContextBound};

    if (!SeveralProps.has_property<sycl_property::buffer::use_host_ptr>()) {
      std::cerr << "Error: property list has no property while should have."
                << std::endl;
      Failed = true;
    }

    if (!SeveralProps.has_property<sycl_property::image::use_host_ptr>()) {
      std::cerr << "Error: property list has no property while should have."
                << std::endl;
      Failed = true;
    }

    try {
      sycl_property::buffer::context_bound ContextBoundRet =
          SeveralProps.get_property<sycl_property::buffer::context_bound>();
      if (SYCLContext != ContextBoundRet.get_context()) {
        std::cerr << "Error: returned SYCL context is not the same that was "
                     "passed to c'tor earlier."
                  << std::endl;
        Failed = true;
      }

    } catch (cl::sycl::invalid_object_error &Error) {
      Error.what();
      std::cerr << "Error: exception was thrown in get_property method."
                << std::endl;
      Failed = true;
    }
  }

  {
    cl::sycl::property_list MemChannelProp{
        sycl_property::buffer::mem_channel(2)};
    if (!MemChannelProp.has_property<sycl_property::buffer::mem_channel>()) {
      std::cerr << "Error: property list has no property while should have."
                << std::endl;
      Failed = true;
    }
    auto Prop =
        MemChannelProp.get_property<sycl_property::buffer::mem_channel>();
    if (Prop.get_channel() != 2) {
      std::cerr << "Error: mem_channel property is not equal to 2."
                << std::endl;
      Failed = true;
    }
  }

  std::cerr << "Test status : " << (Failed ? "FAILED" : "PASSED") << std::endl;

  return Failed;
}
