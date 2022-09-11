#include <catch2/catch.hpp>
#include <execution.hpp>

namespace ex = std::execution;

TEST_CASE("then returns a sender", "[adaptors][then]") {
  auto snd = ex::then(ex::just(), [] {});
  STATIC_REQUIRE(ex::sender<decltype(snd)>);
  (void)snd;
}

