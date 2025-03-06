// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/properties/queue_properties.hpp>

#include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>

// Uncomment to print additional test information
// #define VERBOSE_PRINT
namespace syclexp = sycl::ext::oneapi::experimental;

int main() {

  sycl::device Dev;
  sycl::property_list QProp{sycl::property::queue::in_order()};
  sycl::queue Q(QProp);
  sycl::context Ctx = Q.get_context();
  size_t Width = 8;

  try {

    // Pool properties
    syclexp::property::initial_threshold InitialThreshold(1024);
    syclexp::property::maximum_size MaximumSize(4096);
    syclexp::property::read_only ReadOnly;
    syclexp::property::zero_init ZeroInit;
    sycl::property_list PoolProps{InitialThreshold, MaximumSize, ReadOnly,
                                  ZeroInit};

    // Create pools -- device only
    sycl::usm::alloc Kind = sycl::usm::alloc::device;
    syclexp::memory_pool MemPool1(Ctx, Dev, Kind, PoolProps);
    syclexp::memory_pool MemPool2(Q, sycl::usm::alloc::device);

    // std::hash specialization to ensure `memory_pool` follows common reference
    // semantics
    assert(std::hash<syclexp::memory_pool>{}(MemPool1) !=
           std::hash<syclexp::memory_pool>{}(MemPool2));

    // Copy construct
    syclexp::memory_pool MemPoolCopyConstruct(MemPool1);

    // Copy assign
    syclexp::memory_pool MemPoolCopyAssign = MemPoolCopyConstruct;

    // Move construct
    syclexp::memory_pool MemPoolMoveConstruct(std::move(MemPool2));

    // Move assign
    syclexp::memory_pool MemPoolMoveAssign = std::move(MemPoolMoveConstruct);

    // Equality operators
    assert(MemPoolCopyAssign == MemPool1 && "Pool equality is incorrect!");
    assert(MemPoolMoveAssign != MemPool1 && "Pool inequality is incorrect!");

    // Check pool getters
    assert(Ctx == MemPool1.get_context() &&
           "Stored pool context is incorrect!");
    assert(Dev == MemPool1.get_device() && "Stored pool device is incorrect!");
    assert(Kind == MemPool1.get_alloc_kind() &&
           "Stored pool allocation kind is incorrect!");

    // Check property has-ers/getters
    assert(MemPool1.has_property<syclexp::property::initial_threshold>() &&
           "Pool does not have property when it should!");
    assert(MemPool1.has_property<syclexp::property::maximum_size>() &&
           "Pool does not have property when it should!");
    assert(MemPool1.has_property<syclexp::property::read_only>() &&
           "Pool does not have property when it should!");
    assert(MemPool1.has_property<syclexp::property::zero_init>() &&
           "Pool does not have property when it should!");
    assert(!MemPoolMoveAssign
                .has_property<syclexp::property::initial_threshold>() &&
           "Pool has property when it should not!");
    assert(!MemPoolMoveAssign.has_property<syclexp::property::maximum_size>() &&
           "Pool has property when it should not!");
    assert(!MemPoolMoveAssign.has_property<syclexp::property::read_only>() &&
           "Pool has property when it should not!");
    assert(!MemPoolMoveAssign.has_property<syclexp::property::zero_init>() &&
           "Pool has property when it should not!");

    assert(MemPool1.get_property<syclexp::property::initial_threshold>()
                   .get_initial_threshold() ==
               InitialThreshold.get_initial_threshold() &&
           "Pool property values do not match!");
    assert(MemPool1.get_property<syclexp::property::maximum_size>()
                   .get_maximum_size() == MaximumSize.get_maximum_size() &&
           "Pool property values do not match!");

    size_t ReleaseThresholdGet = MemPool1.get_threshold();
    size_t ReservedSizeCurrent = MemPool1.get_reserved_size_current();
    size_t ReservedSizeHigh = MemPool1.get_reserved_size_high();
    size_t UsedSizeCurrent = MemPool1.get_used_size_current();
    size_t UsedSizeHigh = MemPool1.get_used_size_high();
#ifdef VERBOSE_PRINT
    std::cout << "Memory pool release threshold: " << ReleaseThresholdGet
              << std::endl;
    std::cout << "Memory pool current reserved size: " << ReservedSizeCurrent
              << std::endl;
    std::cout << "Memory pool high reserved size: " << ReservedSizeHigh
              << std::endl;
    std::cout << "Memory pool current used size: " << UsedSizeCurrent
              << std::endl;
    std::cout << "Memory pool high used size: " << UsedSizeHigh << std::endl;
#endif

    // Set new threshold -- then check getter
    size_t NewThreshold = 2048;
    MemPool1.set_new_threshold(NewThreshold);
    ReleaseThresholdGet = MemPool1.get_threshold();
#ifdef VERBOSE_PRINT
    std::cout << "Newly set memory pool release threshold: "
              << ReleaseThresholdGet << std::endl;
#endif

    // Reset high watermarks
    MemPool1.reset_reserved_size_high();
    MemPool1.reset_used_size_high();

    // Pool trimming
    MemPool1.trim_to(1024);

    // Default memory pool
    syclexp::memory_pool DefaultPool =
        Ctx.ext_oneapi_get_default_memory_pool(Dev, sycl::usm::alloc::device);

    // Check default memory pool getter is equal
    syclexp::memory_pool DefaultPoolCopy =
        Ctx.ext_oneapi_get_default_memory_pool(Dev, sycl::usm::alloc::device);
    assert(DefaultPool == DefaultPoolCopy &&
           "Default pool is not equivalent between calls!");

  } catch (sycl::exception &E) {
    std::cerr << "SYCL exception caught! : " << E.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
    return 2;
  }

  std::cout << "Test passed!" << std::endl;
  return 0;
}
