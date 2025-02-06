// REQUIRES: aspect-ext_oneapi_async_memory_alloc

// RUN: %{build} -o %t.out
// RUN: %{run-unfiltered-devices} %t.out

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/properties/queue_properties.hpp>

#include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>

// Uncomment to print additional test information
// #define VERBOSE_PRINT
namespace syclexp = sycl::ext::oneapi::experimental;

int main() {

  sycl::device dev;
  sycl::property_list q_prop{sycl::property::queue::in_order()};
  sycl::queue q(q_prop);
  sycl::context ctx = q.get_context();
  size_t width = 8;

  try {

    // Pool properties
    syclexp::property::initial_threshold initialThreshold(1024);
    syclexp::property::maximum_size maximumSize(4096);
    syclexp::property::read_only readOnly;
    syclexp::property::zero_init zeroInit;
    sycl::property_list poolProps{initialThreshold, maximumSize, readOnly,
                                  zeroInit};

    // Create pools -- device only
    sycl::usm::alloc kind = sycl::usm::alloc::device;
    syclexp::memory_pool memPool1(ctx, dev, kind, poolProps);
    syclexp::memory_pool memPool2(q, sycl::usm::alloc::device);

    // std::hash specialization to ensure `memory_pool` follows common reference
    // semantics
    assert(std::hash<syclexp::memory_pool>{}(memPool1) !=
           std::hash<syclexp::memory_pool>{}(memPool2));

    // Copy construct
    syclexp::memory_pool memPoolCopyConstruct(memPool1);

    // Copy assign
    syclexp::memory_pool memPoolCopyAssign = memPoolCopyConstruct;

    // Move construct
    syclexp::memory_pool memPoolMoveConstruct(std::move(memPool2));

    // Move assign
    syclexp::memory_pool memPoolMoveAssign = std::move(memPoolMoveConstruct);

    // Equality operators
    assert(memPoolCopyAssign == memPool1 && "Pool equality is incorrect!");
    assert(memPoolMoveAssign != memPool1 && "Pool inequality is incorrect!");

    // Check pool getters
    assert(ctx == memPool1.get_context() &&
           "Stored pool context is incorrect!");
    assert(dev == memPool1.get_device() && "Stored pool device is incorrect!");
    assert(kind == memPool1.get_alloc_kind() &&
           "Stored pool allocation kind is incorrect!");

    // Check property has-ers/getters
    assert(memPool1.has_property<syclexp::property::initial_threshold>() &&
           "Pool does not have property when it should!");
    assert(memPool1.has_property<syclexp::property::maximum_size>() &&
           "Pool does not have property when it should!");
    assert(memPool1.has_property<syclexp::property::read_only>() &&
           "Pool does not have property when it should!");
    assert(memPool1.has_property<syclexp::property::zero_init>() &&
           "Pool does not have property when it should!");
    assert(!memPoolMoveAssign
                .has_property<syclexp::property::initial_threshold>() &&
           "Pool has property when it should not!");
    assert(!memPoolMoveAssign.has_property<syclexp::property::maximum_size>() &&
           "Pool has property when it should not!");
    assert(!memPoolMoveAssign.has_property<syclexp::property::read_only>() &&
           "Pool has property when it should not!");
    assert(!memPoolMoveAssign.has_property<syclexp::property::zero_init>() &&
           "Pool has property when it should not!");

    assert(memPool1.get_property<syclexp::property::initial_threshold>()
                   .get_initial_threshold() ==
               initialThreshold.get_initial_threshold() &&
           "Pool property values do not match!");
    assert(memPool1.get_property<syclexp::property::maximum_size>()
                   .get_maximum_size() == maximumSize.get_maximum_size() &&
           "Pool property values do not match!");

    size_t maxSizeGet = memPool1.get_max_size();
    size_t releaseThresholdGet = memPool1.get_threshold();
#ifdef VERBOSE_PRINT
    std::cout << "Memory pool maximum size: " << maxSizeGet << std::endl;
    std::cout << "Memory pool release threshold: " << releaseThresholdGet
              << std::endl;
#endif

    // Set new threshold -- then check getter
    size_t newThreshold = 2048;
    memPool1.set_new_threshold(newThreshold);
    releaseThresholdGet = memPool1.get_threshold();
#ifdef VERBOSE_PRINT
    std::cout << "Newly set memory pool release threshold: "
              << releaseThresholdGet << std::endl;
#endif

    // Default memory pool
    syclexp::memory_pool defaultPool =
        ctx.ext_oneapi_get_default_memory_pool(dev, sycl::usm::alloc::device);

    // Check default memory pool getter is equal
    syclexp::memory_pool defaultPoolCopy =
        ctx.ext_oneapi_get_default_memory_pool(dev, sycl::usm::alloc::device);
    assert(defaultPool == defaultPoolCopy &&
           "Default pool is not equivalent between calls!");

  } catch (sycl::exception e) {
    std::cerr << "SYCL exception caught! : " << e.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
    return 2;
  }

  std::cout << "Test passed!" << std::endl;
  return 0;
}
