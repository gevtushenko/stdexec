#include <catch2/catch.hpp>
#include <execution.hpp>

#include "schedulers/stream.cuh"
#include "common.cuh"

namespace ex = std::execution;
namespace stream = example::cuda::stream;

using example::cuda::is_on_gpu;

TEST_CASE("bulk returns a sender", "[cuda][stream][adaptors][bulk]") {
  auto snd = ex::bulk(ex::schedule(stream::scheduler_t{}), 42, [] {});
  STATIC_REQUIRE(ex::sender<decltype(snd)>);
  (void)snd;
}

TEST_CASE("bulk executes on GPU", "[cuda][stream][adaptors][bulk]") {
  flags_storage_t<4> flags_storage{};
  auto flags = flags_storage.get();

  auto snd = ex::schedule(stream::scheduler_t{}) //
           | ex::bulk(4, [=](int idx) {
               if (is_on_gpu()) {
                 flags.set(idx);
               }
             });
  std::this_thread::sync_wait(std::move(snd));

  REQUIRE(flags_storage.all_set_once());
}

TEST_CASE("bulk forwards values on GPU", "[cuda][stream][adaptors][bulk]") {
  flags_storage_t<1024> flags_storage{};
  auto flags = flags_storage.get();

  auto snd = ex::transfer_just(stream::scheduler_t{}, 42) //
           | ex::bulk(1024, [=](int idx, int val) {
               if (is_on_gpu()) {
                 if (val == 42) {
                   flags.set(idx);
                 }
               }
             });
  std::this_thread::sync_wait(std::move(snd));

  REQUIRE(flags_storage.all_set_once());
}

TEST_CASE("bulk forwards multiple values on GPU", "[cuda][stream][adaptors][bulk]") {
  flags_storage_t<2> flags_storage{};
  auto flags = flags_storage.get();

  auto snd = ex::transfer_just(stream::scheduler_t{}, 42, 4.2) //
           | ex::bulk(2, [=](int idx, int i, double d) {
               if (is_on_gpu()) {
                 if (i == 42 && d == 4.2) {
                   flags.set(idx);
                 }
               }
             });
  const auto [i, d] = std::this_thread::sync_wait(std::move(snd)).value();

  REQUIRE(flags_storage.all_set_once());
  REQUIRE(i == 42);
  REQUIRE(d == 4.2);
}

