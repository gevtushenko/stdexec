#include <nvexec/stream_context.cuh>
#include <stdexec/execution.hpp>

#include <cstdio>

namespace ex = stdexec;

struct bulk_fn
{
  int lbl{};

  __host__ __device__ void operator()(int i) {
    std::printf("B%d: i = %d\n", lbl, i); 
  }
};

int main() {
  nvexec::stream_context stream_ctx{};
  nvexec::stream_scheduler sch = stream_ctx.get_scheduler();

  auto snd = ex::schedule(sch)
           | ex::bulk(4, bulk_fn{0});

// stdexec::sync_wait(std::move(snd));
}

