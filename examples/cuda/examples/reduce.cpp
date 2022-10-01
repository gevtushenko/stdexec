#include <schedulers/stream.cuh>
#include <execution.hpp>

#include <thrust/device_vector.h>

#include <cstdio>

template <class Iterator>
struct simple_range {
  Iterator first;
  Iterator last;
};

template <class Iterator>
auto begin(simple_range<Iterator>& rng) {
  return rng.first;
}

template <class Iterator>
auto end(simple_range<Iterator>& rng) {
  return rng.last;
}

namespace ex = std::execution;
namespace stream = example::cuda::stream;

int main() {
  const int n = 2 * 1024;
  thrust::device_vector<int> input(n, 1);
  int* first = thrust::raw_pointer_cast(input.data());
  int* last  = thrust::raw_pointer_cast(input.data()) + input.size();

  stream::context_t stream_context{};

  auto snd = ex::transfer_just(stream_context.get_scheduler(), simple_range{first, last})
           | stream::reduce();

  auto [result] = std::this_thread::sync_wait(std::move(snd)).value();

  std::cout << "result: " << result << std::endl;
}
