add_subdirectory(xptitest_subscriber)

add_definitions(-DXPTI_ENABLE_INSTRUMENTATION)
add_sycl_unittest(XptiTraceTests OBJECT
    QueueApiFailures.cpp
    NodeCreation.cpp
    QueueIDCheck.cpp
)
target_link_libraries(XptiTraceTests PRIVATE xpti xptitest_subscriber)
