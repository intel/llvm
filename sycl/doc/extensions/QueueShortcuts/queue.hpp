class queue {
public:
  ...
  template <typename KernelName, typename KernelType>
  event single_task(KernelType KernelFunc);

  template <typename KernelName, typename KernelType>
  event single_task(event DepEvent, KernelType KernelFunc);

  template <typename KernelName, typename KernelType>
  event single_task(std::vector<event> DepEvents, KernelType KernelFunc);

  template <typename KernelName, typename KernelType, int Dims>
  event parallel_for(range<Dims> NumWorkItems, KernelType KernelFunc);

  template <typename KernelName, typename KernelType, int Dims>
  event parallel_for(range<Dims> NumWorkItems, event DepEvent,
                     KernelType KernelFunc);

  template <typename KernelName, typename KernelType, int Dims>
  event parallel_for(range<Dims> NumWorkItems, std::vector<event> DepEvents,
                     KernelType KernelFunc);

  template <typename KernelName, typename KernelType, int Dims>
  event parallel_for(range<Dims> NumWorkItems, id<Dims> WorkItemOffset,
                     KernelType KernelFunc);

  template <typename KernelName, typename KernelType, int Dims>
  event parallel_for(range<Dims> NumWorkItems, id<Dims> WorkItemOffset,
                     event DepEvent, KernelType KernelFunc);

  template <typename KernelName, typename KernelType, int Dims>
  event parallel_for(range<Dims> NumWorkItems, id<Dims> WorkItemOffset,
                     std::vector<event> DepEvents, KernelType KernelFunc);

  template <typename KernelName, typename KernelType, int Dims>
  event parallel_for(nd_range<Dims> ExecutionRange, KernelType KernelFunc);

  template <typename KernelName, typename KernelType, int Dims>
  event parallel_for(nd_range<Dims> ExecutionRange, event DepEvent,
                     KernelType KernelFunc);

  template <typename KernelName, typename KernelType, int Dims>
  event parallel_for(nd_range<Dims> ExecutionRange,
                     std::vector<event> DepEvents, KernelType KernelFunc);
};
