#pragma once

__SYCL_INLINE namespace cl {
  namespace sycl {

  namespace detail {

  class SYCLMemObjAllocator {
    class holder_base {
    public:
      virtual ~holder_base() = default;
      virtual void *allocate(std::size_t) = 0;
      virtual void deallocate(void *, std::size_t) = 0;
      virtual void setAlignment(std::size_t) = 0;
      virtual void *getAllocator() = 0;
      virtual std::size_t getValueSize() const = 0;
    };

    template <typename AllocatorT> class holder : public holder_base {
      using sycl_memory_object_allocator = detail::aligned_allocator<char>;

      template <typename T>
      using EnableIfDefaultAllocator =
          enable_if_t<std::is_same<T, sycl_memory_object_allocator>::value>;

      template <typename T>
      using EnableIfNonDefaultAllocator =
          enable_if_t<!std::is_same<T, sycl_memory_object_allocator>::value>;

    public:
      holder(AllocatorT Allocator)
          : MAllocator(Allocator),
            MValueSize(sizeof(typename AllocatorT::value_type)){};
      ~holder() = default;
      virtual void *allocate(std::size_t Count) override {
        return reinterpret_cast<void *>(MAllocator.allocate(Count));
      }
      virtual void deallocate(void *Ptr, std::size_t Count) override {
        MAllocator.deallocate(
            reinterpret_cast<typename AllocatorT::value_type *>(Ptr), Count);
      }

      void setAlignment(std::size_t RequiredAlign) override {
        setAlignImpl(RequiredAlign);
      }

      virtual void *getAllocator() override { return &MAllocator; }

      virtual std::size_t getValueSize() const override { return MValueSize; }

    private:
      template <typename T = AllocatorT>
      EnableIfNonDefaultAllocator<T> setAlignImpl(std::size_t RequiredAlign) {
        // Do nothing in case of user's allocator.
      }

      template <typename T = AllocatorT>
      EnableIfDefaultAllocator<T> setAlignImpl(std::size_t RequiredAlign) {
        MAllocator.setAlignment(std::max<size_t>(RequiredAlign, 64));
      }
      AllocatorT MAllocator;
      std::size_t MValueSize;
    };

  public:
    template <typename AllocatorT>
    SYCLMemObjAllocator(AllocatorT Allocator)
        : MAllocator(std::unique_ptr<holder_base>(
              new holder<AllocatorT>(Allocator))){};

    template <typename AllocatorT>
    SYCLMemObjAllocator()
        : MAllocator(std::unique_ptr<holder_base>(
              new holder<AllocatorT>(AllocatorT()))){};

    void *allocate(std::size_t Count) { return MAllocator->allocate(Count); }

    void deallocate(void *Ptr, std::size_t Count) {
      MAllocator->deallocate(Ptr, Count);
    }

    void setAlignment(std::size_t RequiredAlignment) {
      MAllocator->setAlignment(RequiredAlignment);
    }

    template <typename AllocatorT> AllocatorT getAllocator() {
      return *reinterpret_cast<AllocatorT *>(MAllocator->getAllocator());
    }

    std::size_t getValueSize() const { return MAllocator->getValueSize(); }

  private:
    std::unique_ptr<holder_base> MAllocator;
  };

  } // namespace detail
  } // namespace sycl
}
