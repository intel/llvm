// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// UNSUPPORTED: level_zero_v2_adapter
// UNSUPPORTED-INTENDED: v2 adapter does not support pool statistics.

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/properties/queue_properties.hpp>

#include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#include <sycl/ext/oneapi/experimental/async_alloc/memory_pool.hpp>

// Uncomment to print additional test information
// #define VERBOSE_PRINT
namespace syclexp = sycl::ext::oneapi::experimental;

int main() {

  sycl::device Dev;
  sycl::property_list QProp{sycl::property::queue::in_order()};
  sycl::queue Q(QProp);
  sycl::context Ctx = Q.get_context();

  try {

    // Pool properties
    syclexp::initial_threshold InitialThreshold(1024);
    syclexp::maximum_size MaximumSize(4096);
    syclexp::zero_init ZeroInit{};
    syclexp::properties PoolProps{InitialThreshold, MaximumSize, ZeroInit};

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

    size_t ReleaseThresholdGet = MemPool1.get_threshold();
    size_t ReservedSizeCurrent = MemPool1.get_reserved_size_current();
    size_t UsedSizeCurrent = MemPool1.get_used_size_current();
#ifdef VERBOSE_PRINT
    std::cout << "Memory pool release threshold: " << ReleaseThresholdGet
              << std::endl;
    std::cout << "Memory pool current reserved size: " << ReservedSizeCurrent
              << std::endl;
    std::cout << "Memory pool current used size: " << UsedSizeCurrent
              << std::endl;
#endif

    // Set new threshold -- then check getter
    size_t NewThreshold = 2048;
    MemPool1.increase_threshold_to(NewThreshold);
    ReleaseThresholdGet = MemPool1.get_threshold();
#ifdef VERBOSE_PRINT
    std::cout << "Newly set memory pool release threshold: "
              << ReleaseThresholdGet << std::endl;
#endif

    // Allocate memory to check queries
    void *dummyPtr = syclexp::async_malloc_from_pool(Q, 2048, MemPool1);

    ReservedSizeCurrent = MemPool1.get_reserved_size_current();
    UsedSizeCurrent = MemPool1.get_used_size_current();
#ifdef VERBOSE_PRINT
    std::cout << "Memory pool current reserved size: " << ReservedSizeCurrent
              << std::endl;
    std::cout << "Memory pool current used size: " << UsedSizeCurrent
              << std::endl;
#endif

    // We don't know what the exact sizes of each could be - but they must each
    // be greater than 0
    assert(ReservedSizeCurrent > 0 &&
           "Pool reserved size has not increased despite allocating memory!");
    assert(UsedSizeCurrent > 0 &&
           "Pool used size has not increased despite allocating memory!");

    // Free that allocation and wait to release back to OS
    syclexp::async_free(Q, dummyPtr);
    Q.wait_and_throw();

    // Default memory pool
    syclexp::memory_pool DefaultPool =
        Ctx.ext_oneapi_get_default_memory_pool(Dev, sycl::usm::alloc::device);
    DefaultPool.increase_threshold_to(1024);

    // Check default memory pool getter is equal
    syclexp::memory_pool DefaultPoolCopy =
        Ctx.ext_oneapi_get_default_memory_pool(Dev, sycl::usm::alloc::device);
    assert(DefaultPool == DefaultPoolCopy &&
           "Default pool is not equivalent between calls!");

    // Check equivalent thresholds between the two copies
    assert(DefaultPool.get_threshold() == DefaultPoolCopy.get_threshold() &&
           "Default pool does not have equivalent thresholds between retrieved "
           "copies!");

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
