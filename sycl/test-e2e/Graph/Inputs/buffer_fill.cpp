// Tests adding a Buffer fill operation as a graph node.

#include "../graph_common.hpp"

int main() {

  queue Queue;

  const size_t N = 10;
  const float Pattern = 3.14f;
  std::vector<float> Data(N);
  buffer<float> Buffer{Data};

  const uint64_t PatternI64 = 0x3333333355555555;
  std::vector<uint64_t> DataI64(N);
  buffer<uint64_t> BufferI64{DataI64};

  const uint32_t PatternI32 = 888;
  std::vector<uint32_t> DataI32(N);
  buffer<uint32_t> BufferI32{DataI32};

  const uint16_t PatternI16 = 777;
  std::vector<uint16_t> DataI16(N);
  buffer<uint16_t> BufferI16{DataI16};

  const uint8_t PatternI8 = 33;
  std::vector<uint8_t> DataI8(N);
  buffer<uint8_t> BufferI8{DataI8};

  Buffer.set_write_back(false);
  BufferI64.set_write_back(false);
  BufferI32.set_write_back(false);
  BufferI16.set_write_back(false);
  BufferI8.set_write_back(false);
  {
    exp_ext::command_graph Graph{
        Queue.get_context(),
        Queue.get_device(),
        {exp_ext::property::graph::assume_buffer_outlives_graph{}}};

    add_node(Graph, Queue, [&](handler &CGH) {
      auto Acc = Buffer.get_access(CGH);
      CGH.fill(Acc, Pattern);
    });

    add_node(Graph, Queue, [&](handler &CGH) {
      auto Acc = BufferI64.get_access(CGH);
      CGH.fill(Acc, PatternI64);
    });

    add_node(Graph, Queue, [&](handler &CGH) {
      auto Acc = BufferI32.get_access(CGH);
      CGH.fill(Acc, PatternI32);
    });

    add_node(Graph, Queue, [&](handler &CGH) {
      auto Acc = BufferI16.get_access(CGH);
      CGH.fill(Acc, PatternI16);
    });

    add_node(Graph, Queue, [&](handler &CGH) {
      auto Acc = BufferI8.get_access(CGH);
      CGH.fill(Acc, PatternI8);
    });

    auto ExecGraph = Graph.finalize();

    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); }).wait();
  }
  host_accessor HostData(Buffer);
  host_accessor HostDataI64(BufferI64);
  host_accessor HostDataI32(BufferI32);
  host_accessor HostDataI16(BufferI16);
  host_accessor HostDataI8(BufferI8);
  for (int i = 0; i < N; i++) {
    assert(HostData[i] == Pattern);
    assert(HostDataI64[i] == PatternI64);
    assert(HostDataI32[i] == PatternI32);
    assert(HostDataI16[i] == PatternI16);
    assert(HostDataI8[i] == PatternI8);
  }

  return 0;
}
