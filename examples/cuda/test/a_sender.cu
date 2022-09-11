#include <catch2/catch.hpp>
#include <execution.hpp>

#include "schedulers/detail/common.cuh"
#include "schedulers/stream.cuh"
#include "common.cuh"

namespace ex = std::execution;
namespace stream = example::cuda::stream;

using example::cuda::is_on_gpu;

TEST_CASE("A sender works on GPU", "[cuda][stream][adaptors][a sender]") {
  auto snd = ex::schedule(stream::scheduler_t{}) //
           | ex::then([=]() -> int {
               return is_on_gpu() ? 42 : 0;
             })
           | a_sender([](int val) -> int {
               return is_on_gpu() && (val == 42) ? 1 : 0;
             });
  const auto [result] = std::this_thread::sync_wait(std::move(snd)).value();
  REQUIRE(result == 1);
}

