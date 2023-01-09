add_definitions(-DXPTI_ENABLE_INSTRUMENTATION)
add_sycl_unittest(XptiTraceTests OBJECT
    QueueApiFailures.cpp
)

target_link_libraries(XptiTraceTests PRIVATE xptifw xpti)
