class queue {
public:
  // ...
  template <typename KernelName, typename KernelType>
  event single_task(KernelType KernelFunc);                                   // (1)

  template <typename KernelName, typename KernelType>
  event single_task(event DepEvent, KernelType KernelFunc);                   // (2)

  template <typename KernelName, typename KernelType>
  event single_task(const std::vector<event> &DepEvents,
                    KernelType KernelFunc); // (3)

  template <typename KernelName, typename KernelType, int Dims>
  event parallel_for(range<Dims> NumWorkItems, KernelType KernelFunc);        // (4)

  template <typename KernelName, typename KernelType, int Dims>
  event parallel_for(range<Dims> NumWorkItems, event DepEvent,
                     KernelType KernelFunc);                                  // (5)

  template <typename KernelName, typename KernelType, int Dims>
  event parallel_for(range<Dims> NumWorkItems,
                     const std::vector<event> &DepEvents,
                     KernelType KernelFunc); // (6)

  template <typename KernelName, typename KernelType, int Dims>
  event parallel_for(range<Dims> NumWorkItems, id<Dims> WorkItemOffset,
                     KernelType KernelFunc);                                  // (7)

  template <typename KernelName, typename KernelType, int Dims>
  event parallel_for(range<Dims> NumWorkItems, id<Dims> WorkItemOffset,
                     event DepEvent, KernelType KernelFunc);                  // (8)

  template <typename KernelName, typename KernelType, int Dims>
  event parallel_for(range<Dims> NumWorkItems, id<Dims> WorkItemOffset,
                     const std::vector<event> &DepEvents,
                     KernelType KernelFunc); // (9)

  template <typename KernelName, typename KernelType, int Dims>
  event parallel_for(nd_range<Dims> ExecutionRange, KernelType KernelFunc);   // (10)

  template <typename KernelName, typename KernelType, int Dims>
  event parallel_for(nd_range<Dims> ExecutionRange, event DepEvent,
                     KernelType KernelFunc);                                  // (11)

  template <typename KernelName, typename KernelType, int Dims>
  event parallel_for(nd_range<Dims> ExecutionRange,
                     const std::vector<event> &DepEvents,
                     KernelType KernelFunc); // (12)
};
