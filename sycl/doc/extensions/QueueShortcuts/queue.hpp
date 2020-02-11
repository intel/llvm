class queue {
public:
  ...
  template <typename KernelName, typename KernelType>
  event single_task(KernelType KernelFunc);

  template <typename KernelName, typename KernelType>
  event single_task(event DepEvent, KernelType KernelFunc);

  template <typename KernelName, typename KernelType>
  event single_task(const vector_class<event> &DepEvents,
                    KernelType KernelFunc);

  template <typename KernelName, typename KernelType, int Dims>
  event parallel_for(range<Dims> NumWorkItems, KernelType KernelFunc);

  template <typename KernelName, typename KernelType, int Dims>
  event parallel_for(range<Dims> NumWorkItems, event DepEvent,
                     KernelType KernelFunc);

  template <typename KernelName, typename KernelType, int Dims>
  event parallel_for(range<Dims> NumWorkItems,
                     const vector_class<event> &DepEvents,
                     KernelType KernelFunc);

  template <typename KernelName, typename KernelType, int Dims>
  event parallel_for(range<Dims> NumWorkItems, id<Dims> WorkItemOffset,
                     KernelType KernelFunc);

  template <typename KernelName, typename KernelType, int Dims>
  event parallel_for(range<Dims> NumWorkItems, id<Dims> WorkItemOffset,
                     event DepEvent, KernelType KernelFunc);

  template <typename KernelName, typename KernelType, int Dims>
  event parallel_for(range<Dims> NumWorkItems, id<Dims> WorkItemOffset,
                     const vector_class<event> &DepEvents,
                     KernelType KernelFunc);

  template <typename KernelName, typename KernelType, int Dims>
  event parallel_for(nd_range<Dims> ExecutionRange, KernelType KernelFunc);

  template <typename KernelName, typename KernelType, int Dims>
  event parallel_for(nd_range<Dims> ExecutionRange, event DepEvent,
                     KernelType KernelFunc);

  template <typename KernelName, typename KernelType, int Dims>
  event parallel_for(nd_range<Dims> ExecutionRange,
                     const vector_class<event> &DepEvents,
                     KernelType KernelFunc);
};
