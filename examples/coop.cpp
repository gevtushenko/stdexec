/*
 * Copyright (c) 2021-2022 NVIDIA Corporation
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

#include <stdexec/execution.hpp>
#include <exec/inline_scheduler.hpp>
#include <exec/on.hpp>

#include <barrier>
#include <thread>
#include <vector>

namespace coop {
  struct aux_channel_t {
    template <class _Receiver, class... _As>
        requires stdexec::tag_invocable<aux_channel_t, _Receiver, _As...>
      void operator()(_Receiver&& __rcvr, _As&&... __as) const noexcept {
        static_assert(stdexec::nothrow_tag_invocable<aux_channel_t, _Receiver, _As...>);
        (void) tag_invoke(aux_channel_t{}, (_Receiver&&) __rcvr, (_As&&) __as...);
      }

    friend constexpr bool tag_invoke(stdexec::custom_completion_channel_t, aux_channel_t) noexcept {
      return true;
    }
  };

  inline constexpr aux_channel_t aux_channel{};

  struct context_state {
    std::size_t thread_id_{};
    std::size_t num_threads_{};
    std::barrier<>* barrier_{};
    void** pointers_{};

    void broadcast(void *ptr) noexcept {
      for (std::size_t tid = 0; tid < num_threads_; tid++) {
        pointers_[tid] = ptr;
      }
    }

    template <class T>
      T* ptr() noexcept {
        return static_cast<T*>(pointers_[thread_id_]);
      }
  };

  namespace schedule_from {
    template <class ReceiverId>
      struct receiver_t {
        using Receiver = stdexec::__t<ReceiverId>;
        using Env = stdexec::env_of_t<Receiver>;

        context_state state_;
        Receiver receiver_;

        template <stdexec::__one_of<stdexec::set_error_t,
                                    stdexec::set_stopped_t> Tag,
                  class... As>
        friend void tag_invoke(Tag tag, receiver_t&& self, As&&... as) noexcept {
          if (self.state_.thread_id_ == 0) {
            tag(std::move(self.receiver_), (As&&)as...);
          } else {
            // TODO
            // aux_channel(std::move(self.receiver_), (As&&)as...);
          }
        }

        template <class... As>
        friend void tag_invoke(stdexec::set_value_t, receiver_t&& self, As&&... as) noexcept {
          self.state_.barrier_->arrive_and_wait();

          if (self.state_.thread_id_ == 0) {
            stdexec::set_value(std::move(self.receiver_), (As&&)as...);
          } else {
            aux_channel(std::move(self.receiver_));
          }
        }

        friend stdexec::env_of_t<stdexec::__t<ReceiverId>>
        tag_invoke(stdexec::get_env_t, const receiver_t& self) {
          return stdexec::get_env(self.receiver_);
        }
      };
  }

  template <class Scheduler, class SenderId>
    struct schedule_from_sender_t {
      using Sender = stdexec::__t<SenderId>;

      context_state state_;
      Sender sndr_;

      template <class Receiver>
        using receiver_t = schedule_from::receiver_t<stdexec::__x<Receiver>>;

      template <stdexec::__decays_to<schedule_from_sender_t> Self, stdexec::receiver Receiver>
          requires stdexec::sender_to<stdexec::__member_t<Self, Sender>, receiver_t<Receiver>>
        friend auto tag_invoke(stdexec::connect_t, Self&& self, Receiver&& rcvr)
          -> stdexec::connect_result_t<stdexec::__member_t<Self, Sender>, receiver_t<Receiver>> {
            return stdexec::connect(((Self&&)self).sndr_, receiver_t<Receiver>{self.state_, (Receiver&&)rcvr});
        }

      template <class _Tag>
        friend Scheduler tag_invoke(stdexec::get_completion_scheduler_t<_Tag>, const schedule_from_sender_t& __self) noexcept {
          return {__self.state_};
        }

      template <stdexec::tag_category<stdexec::forwarding_sender_query> _Tag, class... _As>
          requires stdexec::__callable<_Tag, const Sender&, _As...>
        friend auto tag_invoke(_Tag __tag, const schedule_from_sender_t& __self, _As&&... __as)
          noexcept(stdexec::__nothrow_callable<_Tag, const Sender&, _As...>)
          -> stdexec::__call_result_if_t<stdexec::tag_category<_Tag, stdexec::forwarding_sender_query>, _Tag, const Sender&, _As...> {
          return ((_Tag&&) __tag)(__self.sndr_, (_As&&) __as...);
        }

      template <stdexec::__decays_to<schedule_from_sender_t> _Self, class _Env>
        friend auto tag_invoke(stdexec::get_completion_signatures_t, _Self&&, _Env) ->
          stdexec::make_completion_signatures<
            stdexec::__member_t<_Self, Sender>,
            _Env,
            stdexec::completion_signatures<aux_channel_t()>>;

      schedule_from_sender_t(context_state state, Sender sndr)
        : state_(state)
        , sndr_{(Sender&&)sndr} {
      }
    };

  namespace transfer {
    template <class Variant, class ReceiverId>
      struct receiver_t {
        using Receiver = stdexec::__t<ReceiverId>;
        using Env = stdexec::env_of_t<Receiver>;

        context_state state_;
        Receiver receiver_;
        Variant *variant_{};

        template <stdexec::__one_of<stdexec::set_value_t,
                                    stdexec::set_error_t,
                                    stdexec::set_stopped_t> Tag,
                  class... As>
          friend void tag_invoke(Tag tag, receiver_t&& self, As&&... as) noexcept {
            if (self.state_.thread_id_ == 0) {
              using tuple_t = stdexec::__decayed_tuple<Tag, As...>;
              self.variant_->template emplace<tuple_t>(tag, (As&&)as...);
              self.state_.broadcast(self.variant_);
            }
            aux_channel(std::move(self));
          }

        template <class... As>
          friend void tag_invoke(aux_channel_t, receiver_t&& self) noexcept {
            self.state_.barrier_->arrive_and_wait();

            std::visit([&](const auto& tupl) noexcept -> void {
              std::apply([&](auto tag, const auto&... args) noexcept -> void {
                tag(std::move(self.receiver_), args...);
              }, tupl);
            }, *self.state_.template ptr<Variant>());
          }

        friend stdexec::env_of_t<stdexec::__t<ReceiverId>>
        tag_invoke(stdexec::get_env_t, const receiver_t& self) {
          return stdexec::get_env(self.receiver_);
        }
      };

    template <class SenderId, class ReceiverId>
      struct operation_state_t {
        using Sender = stdexec::__t<SenderId>;
        using Receiver = stdexec::__t<ReceiverId>;

        template <class... _Ts>
          using bind_tuples =
            stdexec::__mbind_front_q<
              stdexec::__variant,
              std::tuple<stdexec::set_stopped_t>, // Initial state of the variant is set_stopped
              std::tuple<stdexec::set_error_t, std::exception_ptr>,
              _Ts...>;

        using bound_values_t =
          stdexec::__value_types_of_t<
            Sender,
            stdexec::env_of_t<Receiver>,
            stdexec::__mbind_front_q<stdexec::__decayed_tuple, stdexec::set_value_t>,
            stdexec::__q<bind_tuples>>;

        using Variant =
          stdexec::__error_types_of_t<
            Sender,
            stdexec::env_of_t<Receiver>,
            stdexec::__transform<
              stdexec::__mbind_front_q<stdexec::__decayed_tuple, stdexec::set_error_t>,
              bound_values_t>>;

        using receiver_th = receiver_t<Variant, ReceiverId>;

        Variant variant_;
        stdexec::connect_result_t<Sender, receiver_th> inner_op_;

        friend void tag_invoke(stdexec::start_t, operation_state_t& op) noexcept {
          stdexec::start(op.inner_op_);
        }

        operation_state_t(context_state state, Sender&& sender, Receiver&& receiver)
          : inner_op_{
              stdexec::connect(
                (Sender&&)sender, 
                receiver_th{
                  state,
                  (Receiver&&)receiver,
                  &variant_})} {
        }
      };
  }

  template <class Scheduler, class SenderId>
    struct transfer_sender_t {
      using Sender = stdexec::__t<SenderId>;

      template <class Self, class Receiver>
        using op_state_th = 
          transfer::operation_state_t<
            stdexec::__id<stdexec::__member_t<Self, Sender>>, 
            stdexec::__id<Receiver>>;

      context_state state_;
      Sender sndr_;

      template <stdexec::__decays_to<transfer_sender_t> Self, stdexec::receiver Receiver>
          requires stdexec::sender_to<stdexec::__member_t<Self, Sender>, typename op_state_th<Self, Receiver>::receiver_th>
        friend auto tag_invoke(stdexec::connect_t, Self&& self, Receiver&& rcvr)
          -> op_state_th<Self, Receiver> {
            return op_state_th<Self, Receiver>(
                self.state_, 
                ((Self&&)self).sndr_, 
                (Receiver&&)rcvr);
        }

      template <stdexec::__one_of<stdexec::set_value_t, stdexec::set_stopped_t, stdexec::set_error_t> _Tag>
        friend Scheduler tag_invoke(stdexec::get_completion_scheduler_t<_Tag>, const transfer_sender_t& __self) noexcept {
          return {__self.state_};
        }

      template <stdexec::tag_category<stdexec::forwarding_sender_query> _Tag, class... _As>
          requires stdexec::__callable<_Tag, const Sender&, _As...>
        friend auto tag_invoke(_Tag __tag, const transfer_sender_t& __self, _As&&... __as)
          noexcept(stdexec::__nothrow_callable<_Tag, const Sender&, _As...>)
          -> stdexec::__call_result_if_t<stdexec::tag_category<_Tag, stdexec::forwarding_sender_query>, _Tag, const Sender&, _As...> {
          return ((_Tag&&) __tag)(__self.sndr_, (_As&&) __as...);
        }

      template <stdexec::__decays_to<transfer_sender_t> _Self, class _Env>
        friend auto tag_invoke(stdexec::get_completion_signatures_t, _Self&&, _Env) ->
          stdexec::make_completion_signatures<
            stdexec::__member_t<_Self, Sender>, 
            _Env,
            stdexec::completion_signatures<
              stdexec::set_error_t(const std::exception_ptr&),
              stdexec::set_stopped_t()>>;

      transfer_sender_t(context_state state, Sender sndr)
        : state_(state)
        , sndr_{(Sender&&)sndr} {
      }
    };

  namespace bulk {
    template <class Variant, class Shape, class Fun, class ReceiverId>
      struct receiver_t {
        using Receiver = stdexec::__t<ReceiverId>;
        using Env = stdexec::env_of_t<Receiver>;

        context_state state_;
        Shape shape_;
        Fun fun_;
        Receiver receiver_;
        Variant *variant_{};

        static std::pair<Shape, Shape>
        even_share(Shape n, std::uint32_t rank, std::uint32_t size) noexcept {
          const auto avg_per_thread = n / size;
          const auto n_big_share = avg_per_thread + 1;
          const auto big_shares = n % size;
          const auto is_big_share = rank < big_shares;
          const auto begin = is_big_share ? n_big_share * rank
                                          : n_big_share * big_shares +
                                              (rank - big_shares) * avg_per_thread;
          const auto end = begin + (is_big_share ? n_big_share : avg_per_thread);

          return std::make_pair(begin, end);
        }

        template <stdexec::__one_of<stdexec::set_value_t,
                                    stdexec::set_error_t,
                                    stdexec::set_stopped_t> Tag,
                  class... As>
          friend void tag_invoke(Tag tag, receiver_t&& self, As&&... as) noexcept {
            if (self.state_.thread_id_ == 0) {
              using tuple_t = stdexec::__decayed_tuple<Tag, As...>;
              self.variant_->template emplace<tuple_t>(tag, (As&&)as...);
              self.state_.broadcast(self.variant_);
            }
            aux_channel(std::move(self));
          }

        template <class... As>
          friend void tag_invoke(aux_channel_t, receiver_t&& self) noexcept {
            self.state_.barrier_->arrive_and_wait();

            std::visit([&](const auto& tupl) noexcept -> void {
              std::apply([&](auto tag, const auto&... args) noexcept -> void {
                if constexpr (std::is_same_v<std::decay_t<decltype(tag)>, stdexec::set_value_t>) {
                  auto [begin, end] = 
                    receiver_t::even_share(self.shape_, self.state_.thread_id_, self.state_.num_threads_);

                  for (Shape i = begin; i < end; ++i) {
                    self.fun_(i, args...);
                  }
                }
              }, tupl);
            }, *self.state_.template ptr<Variant>());
            self.state_.barrier_->arrive_and_wait();

            if (self.state_.thread_id_ == 0) {
              std::visit([&](const auto& tupl) noexcept -> void {
                std::apply([&](auto tag, const auto&... args) noexcept -> void {
                  tag(std::move(self.receiver_), args...);
                }, tupl);
              }, *self.variant_);
            } else {
              aux_channel(std::move(self.receiver_));
            }
          }

        friend stdexec::env_of_t<Receiver>
        tag_invoke(stdexec::get_env_t, const receiver_t& self) {
          return stdexec::get_env(self.receiver_);
        }
      };

    template <class SenderId, class ReceiverId, class Shape, class Fun>
      struct operation_state_t {
        using Sender = stdexec::__t<SenderId>;
        using Receiver = stdexec::__t<ReceiverId>;

        template <class... _Ts>
          using bind_tuples =
            stdexec::__mbind_front_q<
              stdexec::__variant,
              std::tuple<stdexec::set_stopped_t>, // Initial state of the variant is set_stopped
              std::tuple<stdexec::set_error_t, std::exception_ptr>,
              _Ts...>;

        using bound_values_t =
          stdexec::__value_types_of_t<
            Sender,
            stdexec::env_of_t<Receiver>,
            stdexec::__mbind_front_q<stdexec::__decayed_tuple, stdexec::set_value_t>,
            stdexec::__q<bind_tuples>>;

        using Variant =
          stdexec::__error_types_of_t<
            Sender,
            stdexec::env_of_t<Receiver>,
            stdexec::__transform<
              stdexec::__mbind_front_q<stdexec::__decayed_tuple, stdexec::set_error_t>,
              bound_values_t>>;

        using receiver_th = receiver_t<Variant, Shape, Fun, ReceiverId>;

        Variant variant_;
        stdexec::connect_result_t<Sender, receiver_th> inner_op_;

        friend void tag_invoke(stdexec::start_t, operation_state_t& op) noexcept {
          stdexec::start(op.inner_op_);
        }

        operation_state_t(context_state state, Shape shape, Fun fun, Sender&& sender, Receiver&& receiver)
          : inner_op_{
              stdexec::connect(
                (Sender&&)sender, 
                receiver_th{
                  state,
                  shape,
                  fun, 
                  (Receiver&&)receiver,
                  &variant_})} {
        }
      };
  }

  template <class SenderId, class Shape, class Fun>
    struct bulk_sender_t {
      using Sender = stdexec::__t<SenderId>;

      template <class Self, class Receiver>
        using op_state_th = 
          bulk::operation_state_t<
            stdexec::__id<stdexec::__member_t<Self, Sender>>, 
            stdexec::__id<Receiver>, 
            Shape, 
            Fun>;

      context_state state_;
      Sender sndr_;
      Shape shape_;
      Fun fun_;

      template <stdexec::__decays_to<bulk_sender_t> Self, stdexec::receiver Receiver>
          requires stdexec::sender_to<stdexec::__member_t<Self, Sender>, Receiver>
        friend auto tag_invoke(stdexec::connect_t, Self&& self, Receiver&& rcvr)
          -> op_state_th<Self, Receiver> {
            return op_state_th<Self, Receiver>(
                self.state_, 
                self.shape_,
                self.fun_,
                ((Self&&)self).sndr_, 
                (Receiver&&)rcvr);
        }

      template <stdexec::tag_category<stdexec::forwarding_sender_query> _Tag, class... _As>
          requires stdexec::__callable<_Tag, const Sender&, _As...>
        friend auto tag_invoke(_Tag __tag, const bulk_sender_t& __self, _As&&... __as)
          noexcept(stdexec::__nothrow_callable<_Tag, const Sender&, _As...>)
          -> stdexec::__call_result_if_t<stdexec::tag_category<_Tag, stdexec::forwarding_sender_query>, _Tag, const Sender&, _As...> {
          return ((_Tag&&) __tag)(__self.sndr_, (_As&&) __as...);
        }

      template <stdexec::__decays_to<bulk_sender_t> _Self, class _Env>
        friend auto tag_invoke(stdexec::get_completion_signatures_t, _Self&&, _Env) ->
          stdexec::make_completion_signatures<
            stdexec::__member_t<_Self, Sender>,
            _Env,
            stdexec::completion_signatures<aux_channel_t()>>;
    };

  struct inline_scheduler {
    context_state state_;

    template <stdexec::sender Sender>
      using schedule_from_sender_th = schedule_from_sender_t<inline_scheduler, stdexec::__x<std::remove_cvref_t<Sender>>>;

    template <stdexec::sender Sender>
      using transfer_sender_th = transfer_sender_t<inline_scheduler, stdexec::__x<std::remove_cvref_t<Sender>>>;

    template <class R_>
      struct __op {
        using R = stdexec::__t<R_>;
        [[no_unique_address]] R rec_;
        friend void tag_invoke(stdexec::start_t, __op& op) noexcept {
          aux_channel((R&&) op.rec_);
        }
      };

    struct __sender {
      context_state state_;

      template <class R>
        friend auto tag_invoke(stdexec::connect_t, __sender, R&& rec)
          noexcept(std::is_nothrow_constructible_v<std::remove_cvref_t<R>, R>)
          -> __op<stdexec::__x<std::remove_cvref_t<R>>> {
          return {(R&&) rec};
        }

      template <stdexec::__one_of<stdexec::set_value_t, stdexec::set_stopped_t, stdexec::set_error_t> _Tag>
        friend inline_scheduler tag_invoke(stdexec::get_completion_scheduler_t<_Tag>, const __sender& __self) noexcept {
          return {__self.state_};
        }

      // TODO Teach `get_completion_scheduler_t` to custom channels ?
      template <stdexec::__decays_to<__sender> _Self, class _Env>
        friend auto tag_invoke(stdexec::get_completion_signatures_t, _Self&&, _Env) ->
          stdexec::completion_signatures<
            stdexec::set_value_t(),
            stdexec::set_error_t(std::exception_ptr),
            stdexec::set_stopped_t(),
            aux_channel_t()>;
    };

    friend __sender tag_invoke(stdexec::schedule_t, const inline_scheduler& self) noexcept {
      return {self.state_};
    }

    template <stdexec::sender S, stdexec::scheduler Sch>
      friend auto
      tag_invoke(stdexec::transfer_t, const inline_scheduler& sch, S&& sndr, Sch&& scheduler) noexcept {
        return stdexec::schedule_from((Sch&&)scheduler, transfer_sender_th<S>(sch.state_, (S&&)sndr));
      }

    template <stdexec::sender S>
      friend schedule_from_sender_th<S>
      tag_invoke(stdexec::schedule_from_t, const inline_scheduler& sch, S&& sndr) noexcept {
        return schedule_from_sender_th<S>(sch.state_, (S&&) sndr);
      }

    template <stdexec::sender Sender, class Shape, class Fun>
      using bulk_sender_th = 
        bulk_sender_t<stdexec::__id<std::decay_t<Sender>>, Shape, Fun>;

    template <stdexec::sender S, std::integral Shape, class Fn>
      friend bulk_sender_th<S, Shape, Fn>
      tag_invoke(stdexec::bulk_t, const inline_scheduler& sch, S&& sndr, Shape shape, Fn fun) noexcept {
        return bulk_sender_th<S, Shape, Fn>{sch.state_, (S&&) sndr, shape, fun};
      }

    friend stdexec::forward_progress_guarantee tag_invoke(
        stdexec::get_forward_progress_guarantee_t,
        const inline_scheduler&) noexcept {
      return stdexec::forward_progress_guarantee::weakly_parallel;
    }

    bool operator==(const inline_scheduler& other) const noexcept {
      return state_.barrier_ == other.state_.barrier_;
    };
  };

  struct inline_context {
    std::size_t num_threads_{1};
    std::barrier<> barrier_;
    std::unique_ptr<void*[]> pointers_;

    inline_context(std::size_t num_threads)
      : num_threads_(num_threads)
      , barrier_(num_threads_)
      , pointers_(new void*[num_threads_]) {
    }

    inline_scheduler get_scheduler(std::size_t thread_id) {
      return {context_state{thread_id, num_threads_, &barrier_, pointers_.get()}};
    }
  };
}

struct empty_env {};

struct recv {
  template <class T, class... Args>
    friend void tag_invoke(T, recv&&, Args&&...) noexcept {}
  friend empty_env tag_invoke(stdexec::get_env_t, const recv&) noexcept { return {}; }
};

int main() {
  const std::size_t num_threads{2};
  coop::inline_context ctx{num_threads};
  std::vector<std::thread> threads{num_threads};

  for (std::size_t thread_id = 0; thread_id < num_threads; thread_id++) {
    threads[thread_id] = std::thread([thread_id, &ctx] {
      auto sch = ctx.get_scheduler(thread_id);
      auto snd = stdexec::just()
               | exec::on(exec::inline_scheduler{},
                          stdexec::then([]() -> int { 
                            std::printf("inline::then sends 42\n"); 
                            return 42;
                          })
                        | stdexec::bulk(2, [](int id, int val) {
                            std::printf("inline::bulk(%d) forwards %d\n", id, val);
                          }))
               | exec::on(sch,
                          stdexec::then([](int val) -> int { 
                            std::printf("coop forwards %d\n", val); 
                            return val;
                          })
                        | stdexec::bulk(2, [](int id, int val) {
                            std::printf("coop::bulk(%d) forwards %d\n", id, val);
                          })
                        )
               | exec::on(exec::inline_scheduler{},
                          stdexec::then([] (int val) { 
                            std::printf("inline receives %d\n", val); 
                          })
                        | stdexec::bulk(2, [](int id) {
                            std::printf("inline::bulk(%d)\n", id);
                          }));
      stdexec::sync_wait(std::move(snd));
    });
  }

  for (auto& thread: threads) {
    thread.join();
  }
}

