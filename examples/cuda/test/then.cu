#include <catch2/catch.hpp>
#include <execution.hpp>

#include "schedulers/stream.cuh"
#include "common.cuh"

namespace ex = std::execution;
namespace stream = example::cuda::stream;

using example::cuda::is_on_gpu;

TEST_CASE("then returns a sender", "[cuda][stream][adaptors][then]") {
  auto snd = ex::then(ex::schedule(stream::scheduler_t{}), [] {});
  STATIC_REQUIRE(ex::sender<decltype(snd)>);
  (void)snd;
}

TEST_CASE("then executes on GPU", "[cuda][stream][adaptors][then]") {
  flags_storage_t flags_storage{};
  auto flags = flags_storage.get();

  auto snd = ex::schedule(stream::scheduler_t{}) //
           | ex::then([=] {
               if (is_on_gpu()) {
                 flags.set();
               }
             });
  std::this_thread::sync_wait(std::move(snd));

  REQUIRE(flags_storage.all_set_once());
}

TEST_CASE("then accepts values on GPU", "[cuda][stream][adaptors][then]") {
  flags_storage_t flags_storage{};
  auto flags = flags_storage.get();

  auto snd = ex::transfer_just(stream::scheduler_t{}, 42) //
           | ex::then([=](int val) {
               if (is_on_gpu()) {
                 if (val == 42) {
                   flags.set();
                 }
               }
             });
  std::this_thread::sync_wait(std::move(snd));

  REQUIRE(flags_storage.all_set_once());
}

TEST_CASE("then accepts multiple values on GPU", "[cuda][stream][adaptors][then]") {
  flags_storage_t flags_storage{};
  auto flags = flags_storage.get();

  auto snd = ex::transfer_just(stream::scheduler_t{}, 42, 4.2) //
           | ex::then([=](int i, double d) {
               if (is_on_gpu()) {
                 if (i == 42 && d == 4.2) {
                   flags.set();
                 }
               }
             });
  std::this_thread::sync_wait(std::move(snd));

  REQUIRE(flags_storage.all_set_once());
}

TEST_CASE("then returns values on GPU", "[cuda][stream][adaptors][then]") {
  auto snd = ex::schedule(stream::scheduler_t{}) //
           | ex::then([=]() -> int {
               if (is_on_gpu()) {
                 return 42;
               }

               return 0;
             });
  const auto [result] = std::this_thread::sync_wait(std::move(snd)).value();

  REQUIRE(result == 42);
}

