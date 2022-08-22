/*
 * Copyright (c) NVIDIA
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
#include "functional.hpp"
#include <type_traits>
#if defined(__GNUC__) && !defined(__clang__)
int main() { return 0; }
#else

// Pull in the reference implementation of P2300:
#include <execution.hpp>
#include "./algorithms/then.hpp"
#include "../examples/schedulers/inline_scheduler.hpp"

#include <cstdio>
#include <iostream>
#include <thread>
#include <vector>
#include <barrier>

namespace ex = std::execution;
namespace stdex = ex;

thread_local int tid{};
constexpr int n_threads = 2;
std::barrier barrier{n_threads};

template <class Fn>
std::thread fork(int tid_, Fn fn) {
  return std::thread([fn, tid_] {
    tid = tid_;
    fn();
  });
}

bool is_main_thread() {
  return tid == 0;
}

template <class T>
void print(T && ) {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
}

namespace {
  namespace coop {
    struct domain { };

    template <bool IsCooperative, class BaseEnvId>
      struct env {
        using BaseEnv = std::__t<BaseEnvId>;

        BaseEnv base_env_{};

        template <std::__none_of<ex::get_completion_signatures_t> _Tag, 
                  class... _As>
            requires std::tag_invocable<_Tag, const BaseEnv&, _As...>
          friend auto tag_invoke(_Tag __tag, const env& __self, _As&&... __as) noexcept
            -> std::tag_invoke_result_t<_Tag, const BaseEnv&, _As...> {
            return ((_Tag&&) __tag)(__self.base_env_, (_As&&) __as...);
          }

        constexpr static bool is_cooperative = IsCooperative;
      };

    struct get_cooperative_op_state_t {
      template <ex::operation_state T>
        requires std::tag_invocable<get_cooperative_op_state_t, T&>
      auto operator()(T& o) const
        -> std::tag_invoke_result_t<get_cooperative_op_state_t, T&> {
        return tag_invoke(get_cooperative_op_state_t{}, o);
      }
    };

    inline constexpr get_cooperative_op_state_t get_cooperative_op_state{};

    struct is_fake_env_t {
      template <class T>
        requires std::tag_invocable<is_fake_env_t, const T&>
      auto operator()(const T& o) const
        -> std::tag_invoke_result_t<is_fake_env_t, const T&> {
        return tag_invoke(is_fake_env_t{}, o);
      }

      template <class T>
      auto operator()(const T&) const noexcept {
        return std::false_type{};
      }
    };

    inline constexpr is_fake_env_t is_fake_env{};

    template <class T>
    concept coop = T::is_cooperative == true;

    template <class T>
    concept non_coop = !coop<T>;

    namespace wrapper 
    {
      template <class EnvT>
      struct receiver {
        EnvT env_;

        template <std::__one_of<ex::set_value_t, ex::set_error_t, ex::set_stopped_t> _Tag, class... _Args>
        friend void tag_invoke(_Tag, const receiver& __self, _Args&&... __args) noexcept {
        }

        friend env<false, std::__x<EnvT>> tag_invoke(ex::get_env_t, const receiver& __self) {
          return env<false, std::__x<EnvT>>{__self.env_};
        }
      };

      template<class OID, class RID>
      struct op_state {
        using O = std::__t<OID>;
        using R = std::__t<RID>;

        O op_state_;
        R r_;

        friend void tag_invoke(ex::start_t, op_state& self) noexcept {
          if (is_main_thread()) {
            ex::start(self.op_state_);
            ex::set_value(std::move(self.r_));
          } else {
            ex::start(get_cooperative_op_state(self.op_state_));
          }
        }

        friend auto tag_invoke(get_cooperative_op_state_t, op_state& self)
          -> std::tag_invoke_result_t<get_cooperative_op_state_t, O&>
        {
          return get_cooperative_op_state(self.op_state_);
        }
      };

      template <class S, class R>
      using op_state_t =
          op_state<
            std::__x<
              ex::connect_result_t<
                S, 
                receiver<ex::env_of_t<R>>
              >
            >, 
            std::__x<R> 
          >;

      template<class SID>
      struct sender {
        using S = std::__t<SID>;

        S s_;

        template <class> using _error = stdex::completion_signatures<>;
        template <class... Ts> using _value = stdex::completion_signatures<stdex::set_value_t(Ts...)>;

        template <class Env>
        friend auto tag_invoke(stdex::get_completion_signatures_t, const sender&, Env)
          -> stdex::make_completion_signatures<
              S&, Env,
              stdex::completion_signatures<stdex::set_error_t(std::exception_ptr)>,
              _value, _error>;

        friend auto tag_invoke(ex::get_descriptor_t, const sender&) noexcept
          -> ex::sender_descriptor_t<sender(S)>;

        template<std::__decays_to<sender> Self, ex::receiver R>
        friend auto tag_invoke(ex::connect_t, Self && self, R&& r)
          -> op_state_t<std::__member_t<Self, S>, R> {
          ex::env_of_t<R> env = ex::get_env(r);
          return op_state_t<std::__member_t<Self, S>, R>{
              ex::connect(std::forward<Self>(self).s_, receiver<ex::env_of_t<R>>{env}),
              std::forward<R>(r)
          };
        }

        constexpr static bool is_cooperative = true;
      };
    }

    namespace unscoped_schedule_from 
    {
      template <class EnvT>
      struct receiver {
        EnvT env_;

        template <std::__one_of<ex::set_value_t, ex::set_error_t, ex::set_stopped_t> _Tag, class... _Args>
        friend void tag_invoke(_Tag, const receiver& __self, _Args&&... __args) noexcept {
        }

        friend env<false, std::__x<EnvT>> tag_invoke(ex::get_env_t, const receiver& __self) {
          return env<false, std::__x<EnvT>>{__self.env_};
        }
      };

      template<class OID, class RID>
      struct op_state {
        using O = std::__t<OID>;
        using R = std::__t<RID>;

        O op_state_;
        R r_;

        friend void tag_invoke(ex::start_t, op_state& self) noexcept {
          ex::start(self.op_state_);
          barrier.arrive_and_wait();

          if (is_main_thread()) {
            ex::set_value(std::move(self.r_));
          }
        }

        friend op_state& tag_invoke(get_cooperative_op_state_t, op_state& self)
        {
          return self;
        }
      };

      template <class S, class R>
      using op_state_t =
          op_state<
            std::__x<
              ex::connect_result_t<
                S, 
                receiver<ex::env_of_t<R>>
              >
            >, 
            std::__x<R> 
          >;

      template<class SID>
      struct sender {
        using S = std::__t<SID>;

        S s_;

        template <class> using _error = stdex::completion_signatures<>;
        template <class... Ts> using _value = stdex::completion_signatures<stdex::set_value_t(Ts...)>;

        template <class Env>
        friend auto tag_invoke(stdex::get_completion_signatures_t, const sender&, Env)
          -> stdex::make_completion_signatures<
              S&, Env,
              stdex::completion_signatures<stdex::set_error_t(std::exception_ptr)>,
              _value, _error>;

        friend auto tag_invoke(ex::get_descriptor_t, const sender&) noexcept
          -> ex::sender_descriptor_t<ex::unscoped_schedule_from_t(S)>;

        template<std::__decays_to<sender> Self, ex::receiver R>
        friend auto tag_invoke(ex::connect_t, Self && self, R&& r)
          -> op_state_t<std::__member_t<Self, S>, R> {
          ex::env_of_t<R> env = ex::get_env(r);
          return op_state_t<std::__member_t<Self, S>, R>{
              ex::connect(std::forward<Self>(self).s_, receiver<ex::env_of_t<R>>{env}),
              std::forward<R>(r)
          };
        }

        constexpr static bool is_cooperative = true;
      };
    }

    namespace unscoped_transfer
    {
      template <class EID>
      struct receiver {
        using EnvT = std::__t<EID>;
        EnvT env_;

        template <std::__one_of<ex::set_value_t, ex::set_error_t, ex::set_stopped_t> _Tag, class... _Args>
        friend void tag_invoke(_Tag, const receiver& __self, _Args&&... __args) noexcept {
        }

        friend env<true, EID> tag_invoke(ex::get_env_t, const receiver& __self) {
          return env<true, EID>{__self.env_};
        }
      };

      template<class OID, class RID>
      struct op_state {
        using O = std::__t<OID>;
        using R = std::__t<RID>;

        O op_state_;
        R r_;

        friend void tag_invoke(ex::start_t, op_state& self) noexcept {
          if (is_main_thread()) {
            ex::start(self.op_state_);
          } else {
            ex::start(get_cooperative_op_state(self.op_state_));
          }

          barrier.arrive_and_wait();
          ex::set_value(std::move(self.r_));
        }

        friend O& tag_invoke(get_cooperative_op_state_t, op_state& self)
        {
          return self.op_state_;
        }
      };

      template <class S, class R>
      using op_state_t =
          op_state<
            std::__x<
              ex::connect_result_t<
                S, 
                receiver<std::__x<ex::env_of_t<R>>>
              >
            >, 
            std::__x<R> 
          >;

      template<class SID, class SchedID>
      struct sender {
        using S = std::__t<SID>;
        using Sched = std::__t<SchedID>;

        S s_;
        Sched sched_;

        template <class> using _error = stdex::completion_signatures<>;
        template <class... Ts> using _value = stdex::completion_signatures<stdex::set_value_t(Ts...)>;

        template <class Env>
        friend auto tag_invoke(stdex::get_completion_signatures_t, const sender&, Env)
          -> stdex::make_completion_signatures<
              S&, Env,
              stdex::completion_signatures<stdex::set_error_t(std::exception_ptr)>,
              _value, _error>;

        friend auto tag_invoke(ex::get_descriptor_t, const sender&) noexcept
          -> ex::sender_descriptor_t<ex::unscoped_transfer_t(S)>;

        template<std::__decays_to<sender> Self, ex::receiver R>
        friend auto tag_invoke(ex::connect_t, Self && self, R&& r)
          -> op_state_t<std::__member_t<Self, S>, R> {
          ex::env_of_t<R> env = ex::get_env(r);
          return op_state_t<std::__member_t<Self, S>, R>{
              ex::connect(std::forward<Self>(self).s_, receiver<std::__x<ex::env_of_t<R>>>{env}),
              std::forward<R>(r)
          };
        }

        constexpr static bool is_cooperative = true;
      };
    }

    struct inline_scheduler {
      template <typename R>
      struct oper {
        R recv_;
        friend void tag_invoke(ex::start_t, oper& self) noexcept {
          ex::set_value((R &&) self.recv_);
        }
      };

      struct sender {
        using completion_signatures = ex::completion_signatures<ex::set_value_t()>;
        using descriptor_t = ex::sender_descriptor_t<ex::schedule_t()>;

        template <typename R>
        friend oper<R> tag_invoke(ex::connect_t, sender self, R&& r) {
          return {(R &&) r};
        }

        template <std::__one_of<ex::set_value_t, ex::set_error_t, ex::set_stopped_t> CPO>
        friend inline_scheduler tag_invoke(ex::get_completion_scheduler_t<CPO>, sender) noexcept {
          return {};
        }

        constexpr static bool is_cooperative = true;
      };

      friend sender tag_invoke(ex::schedule_t, inline_scheduler) { return {}; }
      friend domain tag_invoke(ex::get_domain_t, inline_scheduler) noexcept {
        return {};
      }

      friend bool operator==(inline_scheduler, inline_scheduler) noexcept { return true; }
      friend bool operator!=(inline_scheduler, inline_scheduler) noexcept { return false; }
    };

    ex::sender auto tag_invoke(
      ex::connect_transform_t,
      coop::domain,
      ex::unscoped_schedule_from_t,
      auto&& sched_from,
      auto&& env) {
      auto &&[sched, snd] = sched_from;
      using snd_t = decltype(snd);
      return unscoped_schedule_from::sender<std::__x<snd_t>>{snd};
    }

    ex::sender auto tag_invoke(
      ex::connect_transform_t,
      coop::domain,
      ex::unscoped_transfer_t,
      non_coop auto&& transfer,
      auto&& env) {
      auto &&[sndr, sched] = transfer;
      using sndr_t = decltype(sndr);
      using sched_t = decltype(sched);
      return unscoped_transfer::sender<std::__x<sndr_t>, std::__x<sched_t>>{sndr, sched};
    }

    template <non_coop S, coop E>
    wrapper::sender<std::__x<S>> tag_invoke(
      ex::connect_transform_t,
      coop::domain,
      ex::then_t,
      S&& then,
      E&& env) {
      return wrapper::sender<std::__x<S>>{then};
    }
  } // namespace custom
} // anonymous namespace

struct print_t {
  int val_{};

  void operator()() {
    std::printf("then(%d) in thread %d\n", val_, tid);
  }
};

auto print(int val) {
  return print_t{val};
}

struct inline_scheduler {
  struct Domain {};

  template <typename R>
  struct op_state{
    R recv_;
    friend void tag_invoke(ex::start_t, op_state& self) noexcept {
      ex::set_value((R &&) self.recv_);
    }
  };

  struct my_sender {
    using completion_signatures = ex::completion_signatures<ex::set_value_t()>;
    using descriptor_t = ex::sender_descriptor_t<ex::schedule_t()>;

    template <typename R>
    friend op_state<R> tag_invoke(ex::connect_t, my_sender self, R&& r) {
      return {(R &&) r};
    }

    template <std::__one_of<ex::set_value_t, ex::set_error_t, ex::set_stopped_t> CPO>
    friend inline_scheduler tag_invoke(ex::get_completion_scheduler_t<CPO>, my_sender) noexcept {
      return {};
    }
  };

  friend my_sender tag_invoke(ex::schedule_t, inline_scheduler) { return {}; }
  friend Domain tag_invoke(ex::get_domain_t, inline_scheduler) noexcept {
    return Domain();
  }

  friend bool operator==(inline_scheduler, inline_scheduler) noexcept { return true; }
  friend bool operator!=(inline_scheduler, inline_scheduler) noexcept { return false; }
};

template <bool IsCooperative>
struct test_1 {
  constexpr static bool is_cooperative = IsCooperative;
};

struct test_2 {
};

int main() {
  std::vector<std::thread> threads(n_threads);

  for (int tid_ = 0; tid_ < n_threads; tid_++) {
    threads[tid_] = fork(tid_, [] {
      auto snd = ex::on(inline_scheduler{}, ex::just() | ex::then(print(1)))
               | ex::on(coop::inline_scheduler{}, ex::then(print(2)))
               | ex::on(inline_scheduler{}, ex::then(print(3)));

      std::this_thread::sync_wait(std::move(snd));
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

#endif

