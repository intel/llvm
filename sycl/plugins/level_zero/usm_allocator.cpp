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

// USM allocations are a mimimum of 64KB in size even when a smaller size is
// requested. The implementation distinguishes between allocations of size
// ChunkCutOff (32KB) and those that are larger.
// Allocation requests smaller than ChunkCutoff use chunks taken from a single
// 64KB USM allocation. Thus, for example, for 8-byte allocations, only 1 in
// ~8000 requests results in a new USM allocation. Freeing results only in a
// chunk of a larger 64KB allocation to be marked as available and no real
// return to the system. An allocation is returned to the system only when all
// chunks in a 64KB allocation are freed by the program.
// Allocations larger than ChunkCutOff use a separate USM allocation for each
// request. These are subject to "pooling". That is, when such an allocation is
// freed by the program it is retained in a pool. The pool is available for
// future allocations, which means there are fewer actual USM
// allocations/deallocations.

namespace settings {
// Minimum allocation size that will be requested from the system.
static constexpr size_t SlabMinSize = 64 * 1024; // 64KB

// Allocations <= ChunkCutOff will use chunks from individual slabs.
// Allocations >  ChunkCutOff will be rounded up to a multiple of
// SlabMinSize and allocated to occupy the whole slab.
static constexpr size_t ChunkCutOff = SlabMinSize / 2;
// The largest size which is allocated via the allocator.
// Allocations with size > CutOff bypass the USM allocator and
// go directly to the runtime.
static constexpr size_t CutOff = (size_t)1 << 31; // 2GB

// Unfortunately we cannot deduce the size of the array, so every change
// to the number of buckets should be reflected here.
using BucketsArrayType = std::array<size_t, 53>;

// Generates a list of bucket sizes used by the allocator.
static constexpr BucketsArrayType generateBucketSizes() {

// In order to make bucket sizes constexpr simply write
// them all. There are some restrictions that doesn't
// allow to write this in a nicer way.

// Simple helper to compute power of 2
#define P(n) (1ULL << n)

  BucketsArrayType Sizes = {32,    48,
                            64,    96,
                            128,   192,
                            P(8),  P(8) + P(7),
                            P(9),  P(9) + P(8),
                            P(10), P(10) + P(9),
                            P(11), P(11) + P(10),
                            P(12), P(12) + P(11),
                            P(13), P(13) + P(12),
                            P(14), P(14) + P(13),
                            P(15), P(15) + P(14),
                            P(16), P(16) + P(15),
                            P(17), P(17) + P(16),
                            P(18), P(18) + P(17),
                            P(19), P(19) + P(18),
                            P(20), P(20) + P(19),
                            P(21), P(21) + P(20),
                            P(22), P(22) + P(21),
                            P(23), P(23) + P(22),
                            P(24), P(24) + P(23),
                            P(25), P(25) + P(24),
                            P(26), P(26) + P(25),
                            P(27), P(27) + P(26),
                            P(28), P(28) + P(27),
                            P(29), P(29) + P(28),
                            P(30), P(30) + P(29),
                            CutOff};
#undef P

  return Sizes;
}

static constexpr BucketsArrayType BucketSizes = generateBucketSizes();

// The implementation expects that SlabMinSize is 2^n
static_assert((SlabMinSize & (SlabMinSize - 1)) == 0,
              "SlabMinSize must be a power of 2");

// Protects the capacity checking of the pool.
static sycl::detail::SpinLock PoolLock;

static class SetLimits {
public:
  size_t MaxPoolableSize = 1;
  size_t Capacity = 4;
  size_t MaxPoolSize = 256;
  size_t CurPoolSize = 0;

  SetLimits() {
    // Parse optional parameters of this form (applicable to each context):
    // SYCL_PI_LEVEL_ZERO_USM_ALLOCATOR_SETTINGS=[<MaxPoolableSize>][,[<Capacity>][,[<MaxPoolSize>]]]
    // MaxPoolableSize: Maximum poolable allocation size, specified in MB.
    //                  Default 1MB.
    // Capacity:        Number of pooled allocations in each bucket.
    //                  Default 4.
    // MaxPoolSize:     Maximum size of pool, specified in MB.
    //                  Default 256MB.

    char *PoolParams = getenv("SYCL_PI_LEVEL_ZERO_USM_ALLOCATOR");
    if (PoolParams != nullptr) {
      std::string Params(PoolParams);
      size_t Pos = Params.find(',');
      if (Pos != std::string::npos) {
        if (Pos > 0)
          MaxPoolableSize = std::stoi(Params.substr(0, Pos));
        Params.erase(0, Pos + 1);
        Pos = Params.find(',');
        if (Pos != std::string::npos) {
          if (Pos > 0)
            Capacity = std::stoi(Params.substr(0, Pos));
          Params.erase(0, Pos + 1);
          if (Pos != std::string::npos)
            MaxPoolSize = std::stoi(Params);
        } else {
          Capacity = std::stoi(Params);
        }
      } else
        MaxPoolableSize = std::stoi(Params);
    }
    MaxPoolableSize *= (1 << 20);
    MaxPoolSize *= (1 << 20);
  }
} USMPoolSettings;
} // namespace settings

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

// Represents the allocated memory block of size 'settings::SlabMinSize'
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
  static void regSlab(Slab &);
  static void unregSlab(Slab &);
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
  void *getEnd() const {
    return static_cast<char *>(getPtr()) + settings::SlabMinSize;
  }

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

public:
  Bucket(size_t Sz, USMAllocContext::USMAllocImpl &AllocCtx)
      : Size{Sz}, OwnAllocCtx{AllocCtx} {}

  // Get pointer to allocation that is one piece of an available slab in this
  // bucket.
  void *getChunk();

  // Get pointer to allocation that is a full slab in this bucket.
  void *getSlab();

  size_t getSize() const { return Size; }

  // Free an allocation that is one piece of a slab in this bucket.
  void freeChunk(void *Ptr, Slab &Slab);

  // Free an allocation that is a full slab in this bucket.
  void freeSlab(Slab &Slab);

  SystemMemory &getMemHandle();
  USMAllocContext::USMAllocImpl &getUsmAllocCtx() { return OwnAllocCtx; }

  // Check whether an allocation to be freed can be placed in the pool.
  bool CanPool();

private:
  void onFreeChunk(Slab &);

  // Get a slab to be used for chunked allocations.
  // These slabs are used for allocations <= ChunkCutOff and not pooled.
  decltype(AvailableSlabs.begin()) getAvailSlab();

  // Get a slab that will be used as a whole for a single allocation.
  // These slabs are > ChunkCutOff in size and pooled.
  decltype(AvailableSlabs.begin()) getAvailFullSlab();
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

    Buckets.reserve(settings::BucketSizes.size());

    for (auto &&Size : settings::BucketSizes) {
      Buckets.emplace_back(std::make_unique<Bucket>(Size, *this));
    }
  }

  void *allocate(size_t Size, size_t Alignment);
  void *allocate(size_t Size);
  void deallocate(void *Ptr);

  SystemMemory &getMemHandle() { return *MemHandle; }

  std::shared_timed_mutex &getKnownSlabsMapLock() { return KnownSlabsMapLock; }
  std::unordered_multimap<void *, Slab &> &getKnownSlabs() {
    return KnownSlabs;
  }

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
      Chunks(settings::SlabMinSize / Bkt.getSize()), NumAllocated{0},
      bucket(Bkt), SlabListIter{}, FirstFreeChunkIdx{0} {
  size_t SlabAllocSize = Bkt.getSize();
  if (SlabAllocSize < settings::SlabMinSize)
    SlabAllocSize = settings::SlabMinSize;
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
  void *StartAddr = AlignPtrDown(Slab.getPtr(), settings::SlabMinSize);
  void *EndAddr = static_cast<char *>(StartAddr) + settings::SlabMinSize;

  regSlabByAddr(StartAddr, Slab);
  regSlabByAddr(EndAddr, Slab);
}

void Slab::unregSlab(Slab &Slab) {
  void *StartAddr = AlignPtrDown(Slab.getPtr(), settings::SlabMinSize);
  void *EndAddr = static_cast<char *>(StartAddr) + settings::SlabMinSize;

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

bool Slab::hasAvail() { return NumAllocated != getNumChunks(); }

auto Bucket::getAvailFullSlab() -> decltype(AvailableSlabs.begin()) {
  // Return a slab that will be used for a single allocation.
  if (AvailableSlabs.size() == 0) {
    auto It = AvailableSlabs.insert(AvailableSlabs.begin(),
                                    std::make_unique<Slab>(*this));
    (*It)->setIterator(It);
  } else {
    // If a slab was available in the pool then note that the current pooled
    // size has reduced by the size of this slab.
    settings::USMPoolSettings.CurPoolSize -= Size;
  }

  return AvailableSlabs.begin();
}

void *Bucket::getSlab() {
  std::lock_guard<std::mutex> Lg(BucketLock);

  auto SlabIt = getAvailFullSlab();
  auto *FreeSlab = (*SlabIt)->getSlab();
  auto It =
      UnavailableSlabs.insert(UnavailableSlabs.begin(), std::move(*SlabIt));
  AvailableSlabs.erase(SlabIt);
  (*It)->setIterator(It);
  return FreeSlab;
}

void Bucket::freeSlab(Slab &Slab) {
  std::lock_guard<std::mutex> Lg(BucketLock);
  auto SlabIter = Slab.getIterator();
  assert(SlabIter != UnavailableSlabs.end());
  if (CanPool()) {
    auto It =
        AvailableSlabs.insert(AvailableSlabs.begin(), std::move(*SlabIter));
    UnavailableSlabs.erase(SlabIter);
    (*It)->setIterator(It);
  } else {
    UnavailableSlabs.erase(SlabIter);
  }
}

auto Bucket::getAvailSlab() -> decltype(AvailableSlabs.begin()) {
  if (AvailableSlabs.size() == 0) {
    auto It = AvailableSlabs.insert(AvailableSlabs.begin(),
                                    std::make_unique<Slab>(*this));
    (*It)->setIterator(It);
  }

  return AvailableSlabs.begin();
}

void *Bucket::getChunk() {
  std::lock_guard<std::mutex> Lg(BucketLock);

  auto SlabIt = getAvailSlab();
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

void Bucket::freeChunk(void *Ptr, Slab &Slab) {
  std::lock_guard<std::mutex> Lg(BucketLock);

  Slab.freeChunk(Ptr);

  onFreeChunk(Slab);
}

// The lock must be acquired before calling this method
void Bucket::onFreeChunk(Slab &Slab) {
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

  // If slab has no chunks allocated we could pool it if capacity is available
  // or release it to the system.
  if (Slab.getNumAllocated() == 0) {
    // Pool has no space so release it.
    if (!CanPool()) {
      // Remove the slab when all the chunks from it are deallocated
      // Note: since the slab is stored as unique_ptr, just remove it from
      // the list to remove the list to destroy the object
      auto It = Slab.getIterator();
      assert(It != AvailableSlabs.end());

      AvailableSlabs.erase(It);
    }
  }
}

bool Bucket::CanPool() {
  std::lock_guard<sycl::detail::SpinLock> Lock{settings::PoolLock};
  size_t NewFreeSlabsInBucket = AvailableSlabs.size() + 1;
  if (settings::USMPoolSettings.Capacity >= NewFreeSlabsInBucket) {
    size_t NewPoolSize = settings::USMPoolSettings.CurPoolSize + Size;
    if (settings::USMPoolSettings.MaxPoolSize >= NewPoolSize) {
      settings::USMPoolSettings.CurPoolSize = NewPoolSize;
      return true;
    }
  }
  return false;
}

SystemMemory &Bucket::getMemHandle() { return OwnAllocCtx.getMemHandle(); }

void *USMAllocContext::USMAllocImpl::allocate(size_t Size) {
  if (Size == 0)
    return nullptr;

  if (Size > settings::USMPoolSettings.MaxPoolableSize) {
    return getMemHandle().allocate(Size);
  }

  auto &Bucket = findBucket(Size);
  if (Size > settings::ChunkCutOff) {
    return Bucket.getSlab();
  }

  return Bucket.getChunk();
}

void *USMAllocContext::USMAllocImpl::allocate(size_t Size, size_t Alignment) {
  if (Size == 0)
    return nullptr;

  if (Alignment <= 1)
    return allocate(Size);

  size_t AlignedSize = (Size > 1) ? AlignUp(Size, Alignment) : Alignment;

  // Check if requested allocation size is within pooling limit.
  // If not, just request aligned pointer from the system.
  if (AlignedSize > settings::USMPoolSettings.MaxPoolableSize) {
    return getMemHandle().allocate(Size, Alignment);
  }

  void *Ptr;
  auto &Bucket = findBucket(AlignedSize);
  if (AlignedSize > settings::ChunkCutOff) {
    Ptr = Bucket.getSlab();
  } else {
    Ptr = Bucket.getChunk();
  }
  return AlignPtrUp(Ptr, Alignment);
}

Bucket &USMAllocContext::USMAllocImpl::findBucket(size_t Size) {
  assert(Size <= settings::CutOff && "Unexpected size");

  auto It = std::find_if(
      Buckets.begin(), Buckets.end(),
      [Size](const auto &BucketPtr) { return BucketPtr->getSize() >= Size; });

  assert((It != Buckets.end()) && "Bucket should always exist");

  return *(*It);
}

void USMAllocContext::USMAllocImpl::deallocate(void *Ptr) {
  auto *SlabPtr = AlignPtrDown(Ptr, settings::SlabMinSize);

  // Lock the map on read
  std::shared_lock<std::shared_timed_mutex> Lk(getKnownSlabsMapLock());

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
      if (Bucket.getSize() <= settings::ChunkCutOff) {
        Bucket.freeChunk(Ptr, Slab);
      } else {
        Bucket.freeSlab(Slab);
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

void *USMAllocContext::allocate(size_t size) { return pImpl->allocate(size); }

void *USMAllocContext::allocate(size_t size, size_t alignment) {
  return pImpl->allocate(size, alignment);
}

void USMAllocContext::deallocate(void *ptr) { return pImpl->deallocate(ptr); }

// Define destructor for its usage with unique_ptr
USMAllocContext::~USMAllocContext() = default;
