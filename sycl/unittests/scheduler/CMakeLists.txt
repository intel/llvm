add_sycl_unittest(SchedulerTests OBJECT
    BlockedCommands.cpp
    FailedCommands.cpp
    FinishedCmdCleanup.cpp
    LeafLimit.cpp
    MemObjCommandCleanup.cpp
    CommandsWaitForEvents.cpp
    WaitAfterCleanup.cpp
    LinkedAllocaDependencies.cpp
    CircularBufferExtended.cpp
    utils.cpp
)
