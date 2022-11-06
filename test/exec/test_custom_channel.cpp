/*
 * Copyright (c) 2022 NVIDIA Corporation
 *
 * Licensed under the Apache License Version 2.0 with LLVM Exceptions
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *   https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>
#include <test_common/schedulers.hpp>
#include <test_common/receivers.hpp>

namespace ex = stdexec;

struct custom_channel_t {
  template <class _Receiver, class... _As>
      requires ex::tag_invocable<custom_channel_t, _Receiver, _As...>
    void operator()(_Receiver&& __rcvr, _As&&... __as) const noexcept {
      static_assert(ex::nothrow_tag_invocable<custom_channel_t, _Receiver, _As...>);
      (void) tag_invoke(custom_channel_t{}, (_Receiver&&) __rcvr, (_As&&) __as...);
    }

  friend constexpr bool tag_invoke(ex::custom_completion_channel_t, custom_channel_t) noexcept {
    return true;
  }
};

template <class... Values>
struct custom_just {
  std::tuple<Values...> values_;
  using completion_signatures =
    ex::completion_signatures<
      custom_channel_t(Values...)>;

  template <class Receiver>
  struct operation : immovable {
    std::tuple<Values...> values_;
    Receiver rcvr_;

    friend void tag_invoke(ex::start_t, operation& self) noexcept {
      std::apply(
        [&](Values&... ts) {
          custom_channel_t{}(std::move(self.rcvr_), std::move(ts)...);
        },
        self.values_);
    }
  };

  template <class Receiver>
  friend auto tag_invoke(ex::connect_t, custom_just&& self, Receiver&& rcvr) ->
      operation<std::decay_t<Receiver>> {
    return {{}, std::move(self.values_), std::forward<Receiver>(rcvr)};
  }
};

template <class... Values>
  custom_just(Values...) -> custom_just<Values...>;

template <class _Sender,
          class _Env = ex::no_env>
    requires ex::sender<_Sender, _Env>
  using __custom_types_of_t =
    ex::__gather_sigs_t<custom_channel_t, _Sender, _Env, ex::__q<type_array>, ex::__q<type_array>>;

template <class Expected, typename Env = empty_env, typename S>
inline void check_sends_custom(S snd) {
  using t = __custom_types_of_t<S, Env>;
  static_assert(std::is_same<t, Expected>::value);
}

TEST_CASE("custom channel can appear in the completion signatures", "[custom_channel]") {
  auto snd = custom_just{13};
  static_assert(ex::sender<decltype(snd)>);
  (void)snd;
}

TEST_CASE("custom channel can be advertized", "[custom_channel]") {
  auto snd = custom_just{13};
  check_sends_custom<type_array<type_array<int>>>(snd);
  (void)snd;
}

struct custom_recv_int {
  int expected_{};
  bool &ok_;

  friend void tag_invoke(custom_channel_t, custom_recv_int&& self, int val) noexcept {
    self.ok_ = val == self.expected_;
  }

  friend empty_env tag_invoke(ex::get_env_t, const custom_recv_int&) noexcept { return {}; }
};

TEST_CASE("custom channel can be expected by receivers", "[custom_channel]") {
  bool ok{false};
  auto op = ex::connect(custom_just{13}, custom_recv_int{13, ok});
  ex::start(op);

  REQUIRE(ok);
}

template <class Receiver>
  class fwd_recv : ex::receiver_adaptor<fwd_recv<Receiver>, Receiver> {
    friend ex::receiver_adaptor<fwd_recv<Receiver>, Receiver>;

    template <class... As>
      void set_value(As&&... as) && noexcept {
        std::printf("customize set_value\n");
        ex::set_value_t(((fwd_recv&&) *this).base(), (As&&)as...);
      }

  public:
    fwd_recv(Receiver receiver) 
      : ex::receiver_adaptor<fwd_recv<Receiver>, Receiver>(receiver) {
    }
  };

template <class Receiver>
  fwd_recv(Receiver) -> fwd_recv<Receiver>;

TEST_CASE("custom channels are forwarded by receivers", "[custom_channel]") {
  bool ok{false};
  auto op = ex::connect(custom_just{13}, fwd_recv(custom_recv_int{13, ok}));
  ex::start(op);

  REQUIRE(ok);
}

