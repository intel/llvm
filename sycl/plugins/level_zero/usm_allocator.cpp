//===---------- usm_allocator.cpp - Allocator for USM memory --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "usm_allocator.hpp"
#include <CL/sycl/detail/spinlock.hpp>
#include <iostream>

// USM allocations are a minimum of 4KB/64KB/2MB even when a smaller size is
// requested. The implementation distinguishes between allocations of size
// ChunkCutOff = (minimum-alloc-size / 2) and those that are larger.
// Allocation requests smaller than ChunkCutoff use chunks taken from a single
// USM allocation. Thus, for example, for a 64KB minimum allocation size,
// and 8-byte allocations, only 1 in ~8000 requests results in a new
// USM allocation. Freeing results only in a chunk of a larger allocation
// to be marked as available and no real return to the system.
// An allocation is returned to the system only when all
// chunks in the larger allocation are freed by the program.
// Allocations larger than ChunkCutOff use a separate USM allocation for each
// request. These are subject to "pooling". That is, when such an allocation is
// freed by the program it is retained in a pool. The pool is available for
// future allocations, which means there are fewer actual USM
// allocations/deallocations.

namespace settings {

// Buckets for Host use a minimum of the cache line size of 64 bytes.
// This prevents two separate allocations residing in the same cache line.
// Buckets for Device and Shared allocations will use starting size of 512.
// This is because memory compression on newer GPUs makes the
// minimum granularity 512 bytes instead of 64.
static constexpr size_t MinBucketSize[3] = {64, 512, 512};

// The largest size which is allocated via the allocator.
// Allocations with size > CutOff bypass the USM allocator and
// go directly to the runtime.
static constexpr size_t CutOff = (size_t)1 << 31; // 2GB

// Protects the capacity checking of the pool.
static sycl::detail::SpinLock PoolLock;

static class SetLimits {
public:
  // Minimum allocation size that will be requested from the system.
  // By default this is the minimum allocation size of each memory type.
  // Memory types are host, device, shared.
  size_t SlabMinSize[3] = {64 * 1024, 64 * 1024, 2 * 1024 * 1024};

  // Allocations up to this limit will be subject to chunking/pooling
  size_t MaxPoolableSize[3] = {2 * 1024 * 1024, 4 * 1024 * 1024, 0};

  // When pooling, each bucket will hold a max of 4 unfreed slabs
  size_t Capacity[3] = {4, 4, 0};

  // Maximum memory left unfreed in pool
  size_t MaxPoolSize = 16 * 1024 * 1024;

  size_t CurPoolSize = 0;
  size_t CurPoolSizes[3] = {0, 0, 0};

  size_t EnableBuffers = 1;

  // Whether to print pool usage statistics
  int PoolTrace = 0;

  SetLimits() {
    // Parse optional parameters of this form:
    // SYCL_PI_LEVEL_ZERO_USM_ALLOCATOR=[EnableBuffers][;[MaxPoolSize][;memtypelimits]...]
    //  memtypelimits: [<memtype>:]<limits>
    //  memtype: host|device|shared
    //  limits:  [MaxPoolableSize][,[Capacity][,SlabMinSize]]
    //
    // Without a memory type, the limits are applied to each memory type.
    // Parameters are for each context, except MaxPoolSize, which is overall
    // pool size for all contexts.
    // Duplicate specifications will result in the right-most taking effect.
    //
    // EnableBuffers:   Apply chunking/pooling to SYCL buffers.
    //                  Default 1.
    // MaxPoolSize:     Limit on overall unfreed memory.
    //                  Default 16MB.
    // MaxPoolableSize: Maximum allocation size subject to chunking/pooling.
    //                  Default 2MB host, 4MB device and 0 shared.
    // Capacity:        Maximum number of unfreed allocations in each bucket.
    //                  Default 4.
    // SlabMinSize:     Minimum allocation size requested from USM.
    //                  Default 64KB host and device, 2MB shared.
    //
    // Example of usage:
    // SYCL_PI_LEVEL_ZERO_USM_ALLOCATOR=1;32M;host:1M,4,64K;device:1M,4,64K;shared:0,0,2M

    auto GetValue = [=](std::string &Param, size_t Length, size_t &Setting) {
      size_t Multiplier = 1;
      if (tolower(Param[Length - 1]) == 'k') {
        Length--;
        Multiplier = 1024;
      }
      if (tolower(Param[Length - 1]) == 'm') {
        Length--;
        Multiplier = 1024 * 1024;
      }
      if (tolower(Param[Length - 1]) == 'g') {
        Length--;
        Multiplier = 1024 * 1024 * 1024;
      }
      std::string TheNumber = Param.substr(0, Length);
      if (TheNumber.find_first_not_of("0123456789") == std::string::npos)
        Setting = std::stoi(TheNumber) * Multiplier;
    };

    auto ParamParser = [=](std::string &Params, size_t &Setting,
                           bool &ParamWasSet) {
      bool More;
      if (Params.size() == 0) {
        ParamWasSet = false;
        return false;
      }
      size_t Pos = Params.find(',');
      if (Pos != std::string::npos) {
        if (Pos > 0) {
          GetValue(Params, Pos, Setting);
          ParamWasSet = true;
        }
        Params.erase(0, Pos + 1);
        More = true;
      } else {
        GetValue(Params, Params.size(), Setting);
        ParamWasSet = true;
        More = false;
      }
      return More;
    };

    auto MemParser = [=](std::string &Params, SystemMemory::MemType M) {
      bool ParamWasSet;
      SystemMemory::MemType LM = M;
      if (M == SystemMemory::All)
        LM = SystemMemory::Host;

      bool More = ParamParser(Params, MaxPoolableSize[LM], ParamWasSet);
      if (ParamWasSet && M == SystemMemory::All) {
        MaxPoolableSize[SystemMemory::Shared] =
            MaxPoolableSize[SystemMemory::Device] =
                MaxPoolableSize[SystemMemory::Host];
      }
      if (More) {
        More = ParamParser(Params, Capacity[LM], ParamWasSet);
        if (ParamWasSet && M == SystemMemory::All) {
          Capacity[SystemMemory::Shared] = Capacity[SystemMemory::Device] =
              Capacity[SystemMemory::Host];
        }
      }
      if (More) {
        ParamParser(Params, SlabMinSize[LM], ParamWasSet);
        if (ParamWasSet && M == SystemMemory::All) {
          SlabMinSize[SystemMemory::Shared] =
              SlabMinSize[SystemMemory::Device] =
                  SlabMinSize[SystemMemory::Host];
        }
      }
    };

    auto MemTypeParser = [=](std::string &Params) {
      int Pos = 0;
      SystemMemory::MemType M = SystemMemory::All;
      if (Params.compare(0, 5, "host:") == 0) {
        Pos = 5;
        M = SystemMemory::Host;
      } else if (Params.compare(0, 7, "device:") == 0) {
        Pos = 7;
        M = SystemMemory::Device;
      } else if (Params.compare(0, 7, "shared:") == 0) {
        Pos = 7;
        M = SystemMemory::Shared;
      }
      if (Pos > 0)
        Params.erase(0, Pos);
      MemParser(Params, M);
    };

    // Update pool settings if specified in environment.
    char *PoolParams = getenv("SYCL_PI_LEVEL_ZERO_USM_ALLOCATOR");
    if (PoolParams != nullptr) {
      std::string Params(PoolParams);
      size_t Pos = Params.find(';');
      if (Pos != std::string::npos) {
        if (Pos > 0) {
          GetValue(Params, Pos, EnableBuffers);
        }
        Params.erase(0, Pos + 1);
        size_t Pos = Params.find(';');
        if (Pos != std::string::npos) {
          if (Pos > 0) {
            GetValue(Params, Pos, MaxPoolSize);
          }
          Params.erase(0, Pos + 1);
          do {
            size_t Pos = Params.find(';');
            if (Pos != std::string::npos) {
              if (Pos > 0) {
                std::string MemParams = Params.substr(0, Pos);
                MemTypeParser(MemParams);
              }
              Params.erase(0, Pos + 1);
              if (Params.size() == 0)
                break;
            } else {
              MemTypeParser(Params);
              break;
            }
          } while (true);
        } else {
          GetValue(Params, Params.size(), MaxPoolSize);
        }
      } else {
        GetValue(Params, Params.size(), EnableBuffers);
      }
    }

    char *PoolTraceVal = getenv("SYCL_PI_LEVEL_ZERO_USM_ALLOCATOR_TRACE");
    if (PoolTraceVal != nullptr) {
      PoolTrace = std::atoi(PoolTraceVal);
    }
    if (PoolTrace < 1)
      return;

    std::cout << "USM Pool Settings (Built-in or Adjusted by Environment "
                 "Variable)\n";

    std::cout << std::setw(15) << "Parameter" << std::setw(12) << "Host"
              << std::setw(12) << "Device" << std::setw(12) << "Shared"
              << std::endl;
    std::cout << std::setw(15) << "SlabMinSize" << std::setw(12)
              << SlabMinSize[0] << std::setw(12) << SlabMinSize[1]
              << std::setw(12) << SlabMinSize[2] << std::endl;
    std::cout << std::setw(15) << "MaxPoolableSize" << std::setw(12)
              << MaxPoolableSize[0] << std::setw(12) << MaxPoolableSize[1]
              << std::setw(12) << MaxPoolableSize[2] << std::endl;
    std::cout << std::setw(15) << "Capacity" << std::setw(12) << Capacity[0]
              << std::setw(12) << Capacity[1] << std::setw(12) << Capacity[2]
              << std::endl;
    std::cout << std::setw(15) << "MaxPoolSize" << std::setw(12) << MaxPoolSize
              << std::endl;
    std::cout << std::setw(15) << "EnableBuffers" << std::setw(12)
              << EnableBuffers << std::endl
              << std::endl;
  }
} USMSettings;
} // namespace settings

using namespace settings;

static const char *MemTypeNames[3] = {"Host", "Device", "Shared"};

// Aligns the pointer down to the specified alignment
// (e.g. returns 8 for Size = 13, Alignment = 8)
static void *AlignPtrDown(void *Ptr, const size_t Alignment) {
  return reinterpret_cast<void *>((reinterpret_cast<size_t>(Ptr)) &
                                  (~(Alignment - 1)));
}

// Aligns the pointer up to the specified alignment
// (e.g. returns 16 for Size = 13, Alignment = 8)
static void *AlignPtrUp(void *Ptr, const size_t Alignment) {
  void *AlignedPtr = AlignPtrDown(Ptr, Alignment);
  // Special case when the pointer is already aligned
  if (Ptr == AlignedPtr) {
    return Ptr;
  }
  return static_cast<char *>(AlignedPtr) + Alignment;
}

// Aligns the value up to the specified alignment
// (e.g. returns 16 for Size = 13, Alignment = 8)
static size_t AlignUp(size_t Val, size_t Alignment) {
  assert(Alignment > 0);
  return (Val + Alignment - 1) & (~(Alignment - 1));
}

class Bucket;

// Represents the allocated memory block of size 'SlabMinSize'
// Internally, it splits the memory block into chunks. The number of
// chunks depends of the size of a Bucket which created the Slab.
// The chunks
// Note: Bucket's method are responsible for thread safety of Slab access,
// so no locking happens here.
class Slab {

  // Pointer to the allocated memory of SlabMinSize bytes
  void *MemPtr;

  // Represents the current state of each chunk:
  // if the bit is set then the chunk is allocated
  // the chunk is free for allocation otherwise
  std::vector<bool> Chunks;

  // Total number of allocated chunks at the moment.
  size_t NumAllocated = 0;

  // The bucket which the slab belongs to
  Bucket &bucket;

  using ListIter = std::list<std::unique_ptr<Slab>>::iterator;

  // Store iterator to the corresponding node in avail/unavail list
  // to achieve O(1) removal
  ListIter SlabListIter;

  // Hints where to start search for free chunk in a slab
  size_t FirstFreeChunkIdx = 0;

  // Return the index of the first available chunk, -1 otherwize
  size_t FindFirstAvailableChunkIdx() const;

  // Register/Unregister the slab in the global slab address map.
  void regSlab(Slab &);
  void unregSlab(Slab &);
  static void regSlabByAddr(void *, Slab &);
  static void unregSlabByAddr(void *, Slab &);

public:
  Slab(Bucket &);
  ~Slab();

  void setIterator(ListIter It) { SlabListIter = It; }
  ListIter getIterator() const { return SlabListIter; }

  size_t getNumAllocated() const { return NumAllocated; }

  // Get pointer to allocation that is one piece of this slab.
  void *getChunk();

  // Get pointer to allocation that is this entire slab.
  void *getSlab();

  void *getPtr() const { return MemPtr; }
  void *getEnd() const;

  size_t getChunkSize() const;
  size_t getNumChunks() const { return Chunks.size(); }

  bool hasAvail();

  Bucket &getBucket();
  const Bucket &getBucket() const;

  void freeChunk(void *Ptr);
};

class Bucket {
  const size_t Size;

  // List of slabs which have at least 1 available chunk.
  std::list<std::unique_ptr<Slab>> AvailableSlabs;

  // List of slabs with 0 available chunk.
  std::list<std::unique_ptr<Slab>> UnavailableSlabs;

  // Protects the bucket and all the corresponding slabs
  std::mutex BucketLock;

  // Reference to the allocator context, used access memory allocation
  // routines, slab map and etc.
  USMAllocContext::USMAllocImpl &OwnAllocCtx;

  // Statistics
  size_t allocCount;
  size_t allocPoolCount;
  size_t freeCount;
  size_t currSlabsInUse;
  size_t currSlabsInPool;
  size_t maxSlabsInUse;
  size_t maxSlabsInPool;

public:
  Bucket(size_t Sz, USMAllocContext::USMAllocImpl &AllocCtx)
      : Size{Sz}, OwnAllocCtx{AllocCtx}, allocCount(0), allocPoolCount(0),
        freeCount(0), currSlabsInUse(0), currSlabsInPool(0), maxSlabsInUse(0),
        maxSlabsInPool(0) {}

  // Get pointer to allocation that is one piece of an available slab in this
  // bucket.
  void *getChunk(bool &FromAllocatedSlab);

  // Get pointer to allocation that is a full slab in this bucket.
  void *getSlab(bool &FromPool);

  size_t getSize() const { return Size; }

  // Free an allocation that is one piece of a slab in this bucket.
  void freeChunk(void *Ptr, Slab &Slab, bool &ToPool);

  // Free an allocation that is a full slab in this bucket.
  void freeSlab(Slab &Slab, bool &ToPool);

  SystemMemory &getMemHandle();

  SystemMemory::MemType getMemType();

  USMAllocContext::USMAllocImpl &getUsmAllocCtx() { return OwnAllocCtx; }

  // Check whether an allocation to be freed can be placed in the pool.
  bool CanPool();

  // The minimum allocation size for a slab in this bucket.
  size_t SlabMinSize();

  // The minimum size of a chunk from this bucket's slabs.
  size_t ChunkCutOff();

  // The number of slabs in this bucket that can be in the pool.
  size_t Capacity();

  // The maximum allocation size subject to pooling.
  size_t MaxPoolableSize();

  // Update allocation count
  void countAlloc(bool FromPool);

  // Update free count
  void countFree();

  // Update statistics of Available/Unavailable
  void updateStats(int InUse, int InPool);

  // Print bucket statistics
  void printStats();

private:
  void onFreeChunk(Slab &, bool &ToPool);

  // Get a slab to be used for chunked allocations.
  // These slabs are used for allocations <= ChunkCutOff and not pooled.
  decltype(AvailableSlabs.begin()) getAvailSlab(bool &FromAllocatedSlab);

  // Get a slab that will be used as a whole for a single allocation.
  // These slabs are > ChunkCutOff in size and pooled.
  decltype(AvailableSlabs.begin()) getAvailFullSlab(bool &FromPool);
};

class USMAllocContext::USMAllocImpl {
  // It's important for the map to be destroyed last after buckets and their
  // slabs This is because slab's destructor removes the object from the map.
  std::unordered_multimap<void *, Slab &> KnownSlabs;
  std::shared_timed_mutex KnownSlabsMapLock;

  // Handle to the memory allocation routine
  std::unique_ptr<SystemMemory> MemHandle;

  // Store as unique_ptrs since Bucket is not Movable(because of std::mutex)
  std::vector<std::unique_ptr<Bucket>> Buckets;

public:
  USMAllocImpl(std::unique_ptr<SystemMemory> SystemMemHandle)
      : MemHandle{std::move(SystemMemHandle)} {

    // Generate buckets sized such as: 64, 96, 128, 192, ..., CutOff.
    // Powers of 2 and the value halfway between the powers of 2.
    auto Size1 = MinBucketSize[MemHandle->getMemType()];
    auto Size2 = Size1 + Size1 / 2;
    for (; Size2 < CutOff; Size1 *= 2, Size2 *= 2) {
      Buckets.push_back(std::make_unique<Bucket>(Size1, *this));
      Buckets.push_back(std::make_unique<Bucket>(Size2, *this));
    }
    Buckets.push_back(std::make_unique<Bucket>(CutOff, *this));
  }

  void *allocate(size_t Size, size_t Alignment, bool &FromPool);
  void *allocate(size_t Size, bool &FromPool);
  void deallocate(void *Ptr, bool &ToPool);

  SystemMemory &getMemHandle() { return *MemHandle; }

  std::shared_timed_mutex &getKnownSlabsMapLock() { return KnownSlabsMapLock; }
  std::unordered_multimap<void *, Slab &> &getKnownSlabs() {
    return KnownSlabs;
  }

  size_t SlabMinSize() {
    return USMSettings.SlabMinSize[(*MemHandle).getMemType()];
  };

  void printStats();

private:
  Bucket &findBucket(size_t Size);
};

bool operator==(const Slab &Lhs, const Slab &Rhs) {
  return Lhs.getPtr() == Rhs.getPtr();
}

std::ostream &operator<<(std::ostream &Os, const Slab &Slab) {
  Os << "Slab<" << Slab.getPtr() << ", " << Slab.getEnd() << ", "
     << Slab.getBucket().getSize() << ">";
  return Os;
}

Slab::Slab(Bucket &Bkt)
    : // In case bucket size is not a multiple of SlabMinSize, we would have
      // some padding at the end of the slab.
      Chunks(Bkt.SlabMinSize() / Bkt.getSize()), NumAllocated{0},
      bucket(Bkt), SlabListIter{}, FirstFreeChunkIdx{0} {
  size_t SlabAllocSize = Bkt.getSize();
  if (SlabAllocSize < Bkt.SlabMinSize())
    SlabAllocSize = Bkt.SlabMinSize();
  MemPtr = Bkt.getMemHandle().allocate(SlabAllocSize);
  regSlab(*this);
}

Slab::~Slab() {
  unregSlab(*this);
  bucket.getMemHandle().deallocate(MemPtr);
}

// Return the index of the first available chunk, -1 otherwize
size_t Slab::FindFirstAvailableChunkIdx() const {
  // Use the first free chunk index as a hint for the search.
  auto It = std::find_if(Chunks.begin() + FirstFreeChunkIdx, Chunks.end(),
                         [](auto x) { return !x; });
  if (It != Chunks.end()) {
    return It - Chunks.begin();
  }

  return static_cast<size_t>(-1);
}

void *Slab::getChunk() {
  assert(NumAllocated != Chunks.size());

  const size_t ChunkIdx = FindFirstAvailableChunkIdx();
  // Free chunk must exist, otherwise we would have allocated another slab
  assert(ChunkIdx != (static_cast<size_t>(-1)));

  void *const FreeChunk =
      (static_cast<uint8_t *>(getPtr())) + ChunkIdx * getChunkSize();
  Chunks[ChunkIdx] = true;
  NumAllocated += 1;

  // Use the found index as the next hint
  FirstFreeChunkIdx = ChunkIdx;

  return FreeChunk;
}

void *Slab::getSlab() { return getPtr(); }

Bucket &Slab::getBucket() { return bucket; }
const Bucket &Slab::getBucket() const { return bucket; }

size_t Slab::getChunkSize() const { return bucket.getSize(); }

void Slab::regSlabByAddr(void *Addr, Slab &Slab) {
  auto &Lock = Slab.getBucket().getUsmAllocCtx().getKnownSlabsMapLock();
  auto &Map = Slab.getBucket().getUsmAllocCtx().getKnownSlabs();

  std::lock_guard<std::shared_timed_mutex> Lg(Lock);
  Map.insert({Addr, Slab});
}

void Slab::unregSlabByAddr(void *Addr, Slab &Slab) {
  auto &Lock = Slab.getBucket().getUsmAllocCtx().getKnownSlabsMapLock();
  auto &Map = Slab.getBucket().getUsmAllocCtx().getKnownSlabs();

  std::lock_guard<std::shared_timed_mutex> Lg(Lock);

  auto Slabs = Map.equal_range(Addr);
  // At least the must get the current slab from the map.
  assert(Slabs.first != Slabs.second && "Slab is not found");

  for (auto It = Slabs.first; It != Slabs.second; ++It) {
    if (It->second == Slab) {
      Map.erase(It);
      return;
    }
  }

  assert(false && "Slab is not found");
}

void Slab::regSlab(Slab &Slab) {
  void *StartAddr = AlignPtrDown(Slab.getPtr(), bucket.SlabMinSize());
  void *EndAddr = static_cast<char *>(StartAddr) + bucket.SlabMinSize();

  regSlabByAddr(StartAddr, Slab);
  regSlabByAddr(EndAddr, Slab);
}

void Slab::unregSlab(Slab &Slab) {
  void *StartAddr = AlignPtrDown(Slab.getPtr(), bucket.SlabMinSize());
  void *EndAddr = static_cast<char *>(StartAddr) + bucket.SlabMinSize();

  unregSlabByAddr(StartAddr, Slab);
  unregSlabByAddr(EndAddr, Slab);
}

void Slab::freeChunk(void *Ptr) {
  // This method should be called through bucket(since we might remove the slab
  // as a result), therefore all locks are done on that level.

  // Make sure that we're in the right slab
  assert(Ptr >= getPtr() && Ptr < getEnd());

  // Even if the pointer p was previously aligned, it's still inside the
  // corresponding chunk, so we get the correct index here.
  auto ChunkIdx =
      (static_cast<char *>(Ptr) - static_cast<char *>(MemPtr)) / getChunkSize();

  // Make sure that the chunk was allocated
  assert(Chunks[ChunkIdx] && "double free detected");

  Chunks[ChunkIdx] = false;
  NumAllocated -= 1;

  if (ChunkIdx < FirstFreeChunkIdx)
    FirstFreeChunkIdx = ChunkIdx;
}

void *Slab::getEnd() const {
  return static_cast<char *>(getPtr()) + bucket.SlabMinSize();
}

bool Slab::hasAvail() { return NumAllocated != getNumChunks(); }

auto Bucket::getAvailFullSlab(bool &FromPool)
    -> decltype(AvailableSlabs.begin()) {
  // Return a slab that will be used for a single allocation.
  if (AvailableSlabs.size() == 0) {
    auto It = AvailableSlabs.insert(AvailableSlabs.begin(),
                                    std::make_unique<Slab>(*this));
    (*It)->setIterator(It);
    FromPool = false;
    if (USMSettings.PoolTrace > 1)
      updateStats(1, 0);
  } else {
    // If a slab was available in the pool then note that the current pooled
    // size has reduced by the size of this slab.
    FromPool = true;
    if (USMSettings.PoolTrace > 1) {
      updateStats(1, -1);
      USMSettings.CurPoolSizes[getMemType()] -= Size;
    }
    USMSettings.CurPoolSize -= Size;
  }

  return AvailableSlabs.begin();
}

void *Bucket::getSlab(bool &FromPool) {
  std::lock_guard<std::mutex> Lg(BucketLock);

  auto SlabIt = getAvailFullSlab(FromPool);
  auto *FreeSlab = (*SlabIt)->getSlab();
  auto It =
      UnavailableSlabs.insert(UnavailableSlabs.begin(), std::move(*SlabIt));
  AvailableSlabs.erase(SlabIt);
  (*It)->setIterator(It);
  return FreeSlab;
}

void Bucket::freeSlab(Slab &Slab, bool &ToPool) {
  std::lock_guard<std::mutex> Lg(BucketLock);
  auto SlabIter = Slab.getIterator();
  assert(SlabIter != UnavailableSlabs.end());
  if (CanPool()) {
    auto It =
        AvailableSlabs.insert(AvailableSlabs.begin(), std::move(*SlabIter));
    UnavailableSlabs.erase(SlabIter);
    (*It)->setIterator(It);

    if (USMSettings.PoolTrace > 1) {
      updateStats(-1, 1);
      ToPool = true;
    }
  } else {
    UnavailableSlabs.erase(SlabIter);

    if (USMSettings.PoolTrace > 1) {
      updateStats(-1, 0);
      ToPool = false;
    }
  }
}

auto Bucket::getAvailSlab(bool &FromAllocatedSlab)
    -> decltype(AvailableSlabs.begin()) {

  FromAllocatedSlab = true;
  if (AvailableSlabs.size() == 0) {
    auto It = AvailableSlabs.insert(AvailableSlabs.begin(),
                                    std::make_unique<Slab>(*this));
    (*It)->setIterator(It);

    if (USMSettings.PoolTrace > 1)
      updateStats(1, 0);
    FromAllocatedSlab = false;
  }

  return AvailableSlabs.begin();
}

void *Bucket::getChunk(bool &FromAllocatedSlab) {
  std::lock_guard<std::mutex> Lg(BucketLock);

  auto SlabIt = getAvailSlab(FromAllocatedSlab);
  auto *FreeChunk = (*SlabIt)->getChunk();

  // If the slab is full, move it to unavailable slabs and update its iterator
  if (!((*SlabIt)->hasAvail())) {
    auto It =
        UnavailableSlabs.insert(UnavailableSlabs.begin(), std::move(*SlabIt));
    AvailableSlabs.erase(SlabIt);
    (*It)->setIterator(It);
  }

  return FreeChunk;
}

void Bucket::freeChunk(void *Ptr, Slab &Slab, bool &ToPool) {
  std::lock_guard<std::mutex> Lg(BucketLock);

  Slab.freeChunk(Ptr);

  onFreeChunk(Slab, ToPool);
}

// The lock must be acquired before calling this method
void Bucket::onFreeChunk(Slab &Slab, bool &ToPool) {
  ToPool = true;

  // In case if the slab was previously full and now has 1 available
  // chunk, it should be moved to the list of available slabs
  if (Slab.getNumAllocated() == (Slab.getNumChunks() - 1)) {
    auto SlabIter = Slab.getIterator();
    assert(SlabIter != UnavailableSlabs.end());

    auto It =
        AvailableSlabs.insert(AvailableSlabs.begin(), std::move(*SlabIter));
    UnavailableSlabs.erase(SlabIter);

    (*It)->setIterator(It);
  }

  // Remove the slab when all the chunks from it are deallocated
  // Note: since the slab is stored as unique_ptr, just remove it from
  // the list to remove the list to destroy the object
  if (Slab.getNumAllocated() == 0) {
    auto It = Slab.getIterator();
    assert(It != AvailableSlabs.end());

    AvailableSlabs.erase(It);

    if (USMSettings.PoolTrace > 1)
      updateStats(-1, 0);

    ToPool = false;
  }
}

bool Bucket::CanPool() {
  std::lock_guard<sycl::detail::SpinLock> Lock{PoolLock};
  size_t NewFreeSlabsInBucket = AvailableSlabs.size() + 1;
  if (Capacity() >= NewFreeSlabsInBucket) {
    size_t NewPoolSize = USMSettings.CurPoolSize + Size;
    if (USMSettings.MaxPoolSize >= NewPoolSize) {
      USMSettings.CurPoolSize = NewPoolSize;
      USMSettings.CurPoolSizes[getMemType()] += Size;
      return true;
    }
  }
  return false;
}

SystemMemory &Bucket::getMemHandle() { return OwnAllocCtx.getMemHandle(); }

SystemMemory::MemType Bucket::getMemType() {
  return getMemHandle().getMemType();
}

size_t Bucket::SlabMinSize() { return USMSettings.SlabMinSize[getMemType()]; }

size_t Bucket::Capacity() { return USMSettings.Capacity[getMemType()]; }

size_t Bucket::MaxPoolableSize() {
  return USMSettings.MaxPoolableSize[getMemType()];
}

size_t Bucket::ChunkCutOff() { return SlabMinSize() / 2; }

void Bucket::countAlloc(bool FromPool) {
  ++allocCount;
  if (FromPool)
    ++allocPoolCount;
}

void Bucket::countFree() { ++freeCount; }

void Bucket::updateStats(int InUse, int InPool) {
  currSlabsInUse += InUse;
  maxSlabsInUse = std::max(currSlabsInUse, maxSlabsInUse);
  currSlabsInPool += InPool;
  maxSlabsInPool = std::max(currSlabsInPool, maxSlabsInPool);
}

void Bucket::printStats() {
  if (allocCount) {
    std::cout << std::setw(14) << getSize() << std::setw(12) << allocCount
              << std::setw(12) << freeCount << std::setw(18) << allocPoolCount
              << std::setw(20) << maxSlabsInUse << std::setw(21)
              << maxSlabsInPool << std::endl;
  }
}

// SystemMemory &Bucket::getMemHandle() { return OwnAllocCtx.getMemHandle(); }

void *USMAllocContext::USMAllocImpl::allocate(size_t Size, bool &FromPool) {
  void *Ptr;

  if (Size == 0)
    return nullptr;

  FromPool = false;
  if (Size > USMSettings.MaxPoolableSize[getMemHandle().getMemType()]) {
    return getMemHandle().allocate(Size);
  }

  auto &Bucket = findBucket(Size);

  if (Size > Bucket.ChunkCutOff())
    Ptr = Bucket.getSlab(FromPool);
  else
    Ptr = Bucket.getChunk(FromPool);

  if (USMSettings.PoolTrace > 1)
    Bucket.countAlloc(FromPool);

  return Ptr;
}

void *USMAllocContext::USMAllocImpl::allocate(size_t Size, size_t Alignment,
                                              bool &FromPool) {
  void *Ptr;

  if (Size == 0)
    return nullptr;

  if (Alignment <= 1)
    return allocate(Size, FromPool);

  size_t AlignedSize = (Size > 1) ? AlignUp(Size, Alignment) : Alignment;

  // Check if requested allocation size is within pooling limit.
  // If not, just request aligned pointer from the system.
  FromPool = false;
  if (AlignedSize > USMSettings.MaxPoolableSize[getMemHandle().getMemType()]) {
    return getMemHandle().allocate(Size, Alignment);
  }

  auto &Bucket = findBucket(AlignedSize);

  if (AlignedSize > Bucket.ChunkCutOff()) {
    Ptr = Bucket.getSlab(FromPool);
  } else {
    Ptr = Bucket.getChunk(FromPool);
  }

  if (USMSettings.PoolTrace > 1)
    Bucket.countAlloc(FromPool);

  return AlignPtrUp(Ptr, Alignment);
}

Bucket &USMAllocContext::USMAllocImpl::findBucket(size_t Size) {
  assert(Size <= CutOff && "Unexpected size");

  auto It = std::find_if(
      Buckets.begin(), Buckets.end(),
      [Size](const auto &BucketPtr) { return BucketPtr->getSize() >= Size; });

  assert((It != Buckets.end()) && "Bucket should always exist");

  return *(*It);
}

void USMAllocContext::USMAllocImpl::deallocate(void *Ptr, bool &ToPool) {
  auto *SlabPtr = AlignPtrDown(Ptr, SlabMinSize());

  // Lock the map on read
  std::shared_lock<std::shared_timed_mutex> Lk(getKnownSlabsMapLock());

  ToPool = false;
  auto Slabs = getKnownSlabs().equal_range(SlabPtr);
  if (Slabs.first == Slabs.second) {
    Lk.unlock();
    getMemHandle().deallocate(Ptr);
    return;
  }

  for (auto It = Slabs.first; It != Slabs.second; ++It) {
    // The slab object won't be deleted until it's removed from the map which is
    // protected by the lock, so it's safe to access it here.
    auto &Slab = It->second;
    if (Ptr >= Slab.getPtr() && Ptr < Slab.getEnd()) {
      // Unlock the map before freeing the chunk, it may be locked on write
      // there
      Lk.unlock();
      auto &Bucket = Slab.getBucket();

      if (USMSettings.PoolTrace > 1)
        Bucket.countFree();

      if (Bucket.getSize() <= Bucket.ChunkCutOff()) {
        Bucket.freeChunk(Ptr, Slab, ToPool);
      } else {
        Bucket.freeSlab(Slab, ToPool);
      }

      return;
    }
  }

  Lk.unlock();
  // There is a rare case when we have a pointer from system allocation next
  // to some slab with an entry in the map. So we find a slab
  // but the range checks fail.
  getMemHandle().deallocate(Ptr);
}

USMAllocContext::USMAllocContext(std::unique_ptr<SystemMemory> MemHandle)
    : pImpl(std::make_unique<USMAllocImpl>(std::move(MemHandle))) {}

void *USMAllocContext::allocate(size_t size) {
  bool FromPool;
  auto Ptr = pImpl->allocate(size, FromPool);

  if (USMSettings.PoolTrace > 2) {
    auto MT = pImpl->getMemHandle().getMemType();
    std::cout << "Allocated " << std::setw(8) << size << " " << MemTypeNames[MT]
              << " USM bytes from " << (FromPool ? "Pool" : "USM") << " ->"
              << Ptr << std::endl;
  }
  return Ptr;
}

void *USMAllocContext::allocate(size_t size, size_t alignment) {
  bool FromPool;
  auto Ptr = pImpl->allocate(size, alignment, FromPool);

  if (USMSettings.PoolTrace > 2) {
    auto MT = pImpl->getMemHandle().getMemType();
    std::cout << "Allocated " << std::setw(8) << size << " " << MemTypeNames[MT]
              << " USM bytes aligned at " << alignment << " from "
              << (FromPool ? "Pool" : "USM") << " ->" << Ptr << std::endl;
  }
  return Ptr;
}

void USMAllocContext::deallocate(void *ptr) {
  bool ToPool;
  pImpl->deallocate(ptr, ToPool);

  if (USMSettings.PoolTrace > 2) {
    auto MT = pImpl->getMemHandle().getMemType();
    std::cout << "Freed " << MemTypeNames[MT] << " USM " << ptr << " to "
              << (ToPool ? "Pool" : "USM") << ", Current total pool size "
              << USMSettings.CurPoolSize << ", Current pool sizes ["
              << USMSettings.CurPoolSizes[SystemMemory::Host] << ", "
              << USMSettings.CurPoolSizes[SystemMemory::Device] << ", "
              << USMSettings.CurPoolSizes[SystemMemory::Shared] << "]\n";
  }
  return;
}

// Define destructor for its usage with unique_ptr
USMAllocContext::~USMAllocContext() {
  if (USMSettings.PoolTrace > 1) {
    auto Label = "Shared";
    if (pImpl->getMemHandle().getMemType() == SystemMemory::Host)
      Label = "Host";
    if (pImpl->getMemHandle().getMemType() == SystemMemory::Device)
      Label = "Device";
    std::cout << Label << " memory statistics\n";
    pImpl->printStats();
    std::cout << "Current Pool Size " << USMSettings.CurPoolSize << std::endl;
  }
}

void USMAllocContext::USMAllocImpl::printStats() {
  std::cout << std::setw(14) << "Bucket Size" << std::setw(12) << "Allocs"
            << std::setw(12) << "Frees" << std::setw(18) << "Allocs From Pool"
            << std::setw(20) << "Peak Slabs In Use" << std::setw(21)
            << "Peak Slabs in Pool" << std::endl;
  for (auto &B : Buckets) {
    (*B).printStats();
  }
}

bool enableBufferPooling() { return USMSettings.EnableBuffers; }
