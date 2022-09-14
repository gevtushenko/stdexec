#include <catch2/catch.hpp>
#include <execution.hpp>

#include "schedulers/detail/common.cuh"
#include "schedulers/stream.cuh"
#include "common.cuh"

namespace ex = std::execution;
namespace stream = example::cuda::stream;

using example::cuda::is_on_gpu;

TEST_CASE("A sender works on GPU", "[cuda][stream][adaptors][a sender]") {
  flags_storage_t flag_storage;
  auto flag = flag_storage.get();

  auto snd = ex::schedule(stream::scheduler_t{}) //
           | ex::then([=]() -> int {
               return is_on_gpu() ? 42 : 0;
             })
           | a_sender([flag](int val) {
               if (is_on_gpu() && (val == 42)) {
                 flag.set();
               }
             });

  std::this_thread::sync_wait(std::move(snd));
  REQUIRE(flag_storage.all_set_once());
}

