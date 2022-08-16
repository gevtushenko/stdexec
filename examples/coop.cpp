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
#if defined(__GNUC__) && !defined(__clang__)
int main() { return 0; }
#else

// Pull in the reference implementation of P2300:
#include <execution.hpp>
#include "./algorithms/then.hpp"
#include "../examples/schedulers/inline_scheduler.hpp"

#include <cstdio>
#include <iostream>

namespace ex = std::execution;
namespace stdex = ex;

template <class T>
void print(T && ) {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
}

struct empty_env {};

struct recv0 {
  friend void tag_invoke(ex::set_value_t, recv0&&) noexcept {}
  friend void tag_invoke(ex::set_stopped_t, recv0&&) noexcept {}
  friend void tag_invoke(ex::set_error_t, recv0&&, std::exception_ptr) noexcept {}
  friend empty_env tag_invoke(ex::get_env_t, const recv0&) noexcept { return {}; }
};

// TODO:
// - why accept both sender and a tag in the `connect_transform` if the sender gets passed anyway
// - is receiver == environment? what if receiver forwards environment?
// - is domain always forwarded? 
// - can't pipe senders inside new `ex::on`
// - can domain be undefined for a scheduler? don't see this requirements in the concept
// - can I wrap N senders instead of one?
// - transform will work only inside on?

namespace {
  namespace coop {
    struct domain { };

    struct is_cooperative_t {
      template <class T>
        requires std::tag_invocable<is_cooperative_t, const T&>
      auto operator()(const T& o) const
        -> std::tag_invoke_result_t<is_cooperative_t, const T&> {
        return tag_invoke(is_cooperative_t{}, o);
      }

      template <class T>
      auto operator()(const T&) const noexcept {
        return std::false_type{};
      }
    };

    inline constexpr is_cooperative_t is_cooperative{};

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
    concept fake_env = requires(const T& o) {
      { is_fake_env(o) } -> std::same_as<std::true_type>;
      { is_cooperative(o) } -> std::same_as<std::true_type>;
    };

    template <class T>
    concept coop = requires(const T& o) {
      { is_cooperative(o) } -> std::same_as<std::true_type>;
    };

    template <class T>
    concept non_coop = !coop<T>;

    template <typename R>
    struct oper {
      R recv_;
      friend void tag_invoke(ex::start_t, oper& self) noexcept {
        ex::set_value((R &&) self.recv_);
      }

      friend auto tag_invoke(is_cooperative_t, const oper&) {
        return std::true_type{};
      }
    };

    template<class RID>
    struct fake_receiver : stdex::receiver_adaptor<fake_receiver<RID>> {
      using R = std::__t<RID>;

      R recv_;

      R&& base() && noexcept { return (R&&) recv_; }
      const R& base() const & noexcept { return recv_; }

      void set_value() && noexcept {
        std::printf("fake_receiver::set_value\n");
        ex::set_value(std::move(recv_));
      }

      auto get_env() const {
        return ex::make_env(
            ex::get_env(recv_), 
            ex::with(is_cooperative, std::true_type{}),
            ex::with(is_fake_env, std::true_type{}));
      }

      friend auto tag_invoke(is_cooperative_t, const fake_receiver&) {
        return std::true_type{};
      }

      explicit fake_receiver(R recv) : recv_(recv) {}
    };

    namespace wrapper 
    {
      // pass through all customizations except set_error, which retries the operation.
      template<class RID>
      struct receiver : stdex::receiver_adaptor<receiver<RID>> {
        using R = std::__t<RID>;

        R recv_;

        R&& base() && noexcept { return (R&&) recv_; }
        const R& base() const & noexcept { return recv_; }

        void set_value() && noexcept {
          std::printf("wrapper::receiver::set_value\n");
          ex::set_value(std::move(recv_));
        }

        auto get_env() const {
          return ex::make_env(ex::get_env(base()), ex::with(is_cooperative, std::false_type{}));
        }

        friend auto tag_invoke(is_cooperative_t, const receiver&) {
          return std::false_type{};
        }

        explicit receiver(R recv) : recv_(recv) {}
      };

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
          -> ex::connect_result_t<std::__member_t<Self, S>, receiver<std::__x<R>>> {
          // Can't connect this because it provokes recursion.
          return ex::connect(std::forward<Self>(self).s_, receiver<std::__x<R>>{r});
        }

        friend auto tag_invoke(is_cooperative_t, const sender&) {
          return std::true_type{};
        }
      };
    }

    namespace unscoped_schedule_from 
    {
      template<class RID>
      struct receiver : stdex::receiver_adaptor<receiver<RID>> {
        using R = std::__t<RID>;

        R recv_;

        R&& base() && noexcept { return (R&&) recv_; }
        const R& base() const & noexcept { return recv_; }

        void set_value() && noexcept {
          std::printf("unscoped_schedule_from::receiver::set_value\n");
          ex::set_value(std::move(recv_));
        }

        auto get_env() const {
          return ex::make_env(ex::get_env(recv_), ex::with(is_cooperative, std::false_type{}));
        }

        friend auto tag_invoke(is_cooperative_t, const receiver&) {
          return std::true_type{};
        }

        explicit receiver(R recv) : recv_(recv) {}
      };

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
        friend auto tag_invoke(ex::connect_t, Self&& self, R&& r)
          -> ex::connect_result_t<std::__member_t<Self, S>, receiver<std::__x<R>>> {
          return ex::connect(std::forward<Self>(self).s_, receiver<std::__x<R>>{std::forward<R>(r)});
        }

        friend auto tag_invoke(is_cooperative_t, const sender&) {
          return std::true_type{};
        }
      };
    }

    namespace unscoped_transfer
    {
      template<class RID>
      struct receiver : stdex::receiver_adaptor<receiver<RID>> {
        using R = std::__t<RID>;

        R recv_;

        R&& base() && noexcept { return (R&&) recv_; }
        const R& base() const & noexcept { return recv_; }

        void set_value() && noexcept {
          std::printf("unscoped_transfer::receiver::set_value\n");
          ex::set_value(std::move(recv_));
        }

        auto get_env() const {
          return ex::make_env(ex::get_env(recv_), ex::with(is_cooperative, std::true_type{}));
        }

        friend auto tag_invoke(is_cooperative_t, const receiver&) {
          return std::true_type{};
        }

        explicit receiver(R recv) : recv_(recv) {}
      };

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
        friend auto tag_invoke(stdex::connect_t, Self&& self, R&& r) 
          -> ex::connect_result_t<std::__member_t<Self, S>, receiver<std::__x<R>>> {
          return ex::connect(
              std::forward<Self>(self).s_,
              receiver<std::__x<R>>{r});
        }

        friend auto tag_invoke(is_cooperative_t, const sender&) {
          return std::true_type{};
        }
      };
    }

    struct inline_scheduler {
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

        friend auto tag_invoke(is_cooperative_t, const sender&) {
          return std::true_type{};
        }
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
      auto&& transfer,
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

    /*
    template <coop S, fake_env E>
    ex::sender auto tag_invoke(
      ex::connect_transform_t,
      coop::domain,
      ex::then_t,
      S&& then,
      E&& env) {

      auto&& [sndr, fn] = then;

      std::printf("wrap fake it:\n\t");
      fn();

      // using sndr_t = decltype(sndr);
      return wrapper::sender<std::__x<S>>{then};
    }
    */

    /*
    template <non_coop S, class T, coop E>
    ex::sender auto tag_invoke(
      ex::connect_transform_t,
      coop::domain,
      T,
      S&& s,
      E&& env) {
      return ex::just();
    }
    */
  } // namespace custom
} // anonymous namespace

struct print_t {
  int val_{};

  void operator()() {
    std::printf("then(%d)\n", val_);
  }
};

auto print(int val) {
  return print_t{val};
}

struct inline_scheduler {
  struct Domain {};

  template <typename R>
  struct oper {
    R recv_;
    friend void tag_invoke(ex::start_t, oper& self) noexcept {
      std::printf("inline_scheduler::oper::start\n");
      ex::set_value((R &&) self.recv_);
    }
  };

  struct my_sender {
    using completion_signatures = ex::completion_signatures<ex::set_value_t()>;
    using descriptor_t = ex::sender_descriptor_t<ex::schedule_t()>;

    template <typename R>
    friend oper<R> tag_invoke(ex::connect_t, my_sender self, R&& r) {
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

int main() {
  auto snd = ex::on(inline_scheduler{}, ex::just() | ex::then(print(1)))
           | ex::on(coop::inline_scheduler{}, ex::then(print(2)))
           | ex::on(inline_scheduler{}, ex::then(print(3)));

  std::this_thread::sync_wait(std::move(snd));
}

#endif

