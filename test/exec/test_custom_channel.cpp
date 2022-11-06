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

    friend void tag_invoke(ex::start_t, operation& self) noexcept try {
      std::apply(
        [&](Values&... ts) {
          ex::set_value(std::move(self.rcvr_), std::move(ts)...);
        },
        self.values_);
    } catch(...) {
      ex::set_error(std::move(self.rcvr_), std::current_exception());
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

TEST_CASE("custom channel can be advertized", "[custom_channel]") {
  auto snd = custom_just{13};
  check_sends_custom<type_array<type_array<int>>>(snd);
  (void)snd;
}

