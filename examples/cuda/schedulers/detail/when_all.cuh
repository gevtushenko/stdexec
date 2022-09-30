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
#pragma once

#include <execution.hpp>
#include <type_traits>

#include "common.cuh"
#include "queue.cuh"

namespace example::cuda::stream {

namespace when_all {

enum state_t { started, error, stopped };

struct on_stop_requested {
  std::in_place_stop_source& stop_source_;
  void operator()() noexcept {
    stop_source_.request_stop();
  }
};

template <class Env>
  using env_t =
    std::execution::make_env_t<Env, std::execution::with_t<std::execution::get_stop_token_t, std::in_place_stop_token>>;

template <class...>
  using swallow_values = std::execution::completion_signatures<>;

template <class Env, class... Senders>
  struct traits {
    using __t = std::execution::dependent_completion_signatures<Env>;
  };

template <class Env, class... Senders>
    requires ((std::__v<std::execution::__count_of<std::execution::set_value_t, Senders, Env>> <= 1) &&...)
  struct traits<Env, Senders...> {
    using non_values =
      std::execution::__concat_completion_signatures_t<
        std::execution::completion_signatures<
          std::execution::set_error_t(std::exception_ptr),
          std::execution::set_stopped_t()>,
        std::execution::make_completion_signatures<
          Senders,
          Env,
          std::execution::completion_signatures<>,
          swallow_values>...>;
    using values =
      std::__minvoke<
        std::__concat<std::__qf<std::execution::set_value_t>>,
        std::execution::__value_types_of_t<
          Senders,
          Env,
          std::__q<std::__types>,
          std::__single_or<std::__types<>>>...>;
    using __t =
      std::__if_c<
        (std::execution::__sends<std::execution::set_value_t, Senders, Env> &&...),
        std::__minvoke2<
          std::__push_back<std::__q<std::execution::completion_signatures>>, non_values, values>,
        non_values>;
  };
}

template <class... SenderIds>
  struct when_all_sender_t {
    template <class... Sndrs>
      explicit when_all_sender_t(Sndrs&&... __sndrs)
        : sndrs_((Sndrs&&) __sndrs...)
      {}

   private:
    template <class CvrefEnv>
      using completion_sigs =
        std::__t<when_all::traits<
          when_all::env_t<std::remove_cvref_t<CvrefEnv>>,
          std::__member_t<CvrefEnv, std::__t<SenderIds>>...>>;

    template <class Traits>
      using sends_values =
        std::__bool<std::__v<typename Traits::template
          __gather_sigs<std::execution::set_value_t, std::__mconst<int>, std::__mcount>> != 0>;

    template <class CvrefReceiverId>
      struct operation_t;

    template <class CvrefReceiverId, std::size_t Index>
      struct receiver_t
        : std::execution::receiver_adaptor<receiver_t<CvrefReceiverId, Index>>
        , receiver_base_t {
        using WhenAll = std::__member_t<CvrefReceiverId, when_all_sender_t>;
        using Receiver = std::__t<std::decay_t<CvrefReceiverId>>;
        using SenderId = example::cuda::detail::nth_type<Index, SenderIds...>;
        using Traits =
          completion_sigs<
            std::__member_t<CvrefReceiverId, std::execution::env_of_t<Receiver>>>;

        Receiver&& base() && noexcept {
          return (Receiver&&) op_state_->recvr_;
        }
        const Receiver& base() const & noexcept {
          return op_state_->recvr_;
        }
        template <class Error>
          void set_error(Error&& err, when_all::state_t expected) noexcept {
            // TODO: _What memory orderings are actually needed here?
            if (op_state_->state_.compare_exchange_strong(expected, when_all::error)) {
              op_state_->stop_source_.request_stop();
              // We won the race, free to write the error into the operation
              // state without worry.
              try {
                op_state_->errors_.template emplace<std::decay_t<Error>>((Error&&) err);
              } catch(...) {
                op_state_->errors_.template emplace<std::exception_ptr>(std::current_exception());
              }
            }
            op_state_->arrive();
          }
        template <class... Values>
          void set_value(Values&&... vals) && noexcept {
            if constexpr (sends_values<Traits>::value) {
              // We only need to bother recording the completion values
              // if we're not already in the "error" or "stopped" state.
              if (op_state_->state_ == when_all::started) {
                op_state_->template store_values<Index>((Values&&)vals...);
              }
            }
            op_state_->arrive();
          }
        template <class Error>
            requires std::tag_invocable<std::execution::set_error_t, Receiver, Error>
          void set_error(Error&& err) && noexcept {
            set_error((Error&&) err, when_all::started);
          }
        void set_stopped() && noexcept {
          when_all::state_t expected = when_all::started;
          // Transition to the "stopped" state if and only if we're in the
          // "started" state. (If this fails, it's because we're in an
          // error state, which trumps cancellation.)
          if (op_state_->state_.compare_exchange_strong(expected, when_all::stopped)) {
            op_state_->stop_source_.request_stop();
          }
          op_state_->arrive();
        }
        auto get_env() const
          -> std::execution::make_env_t<std::execution::env_of_t<Receiver>, std::execution::with_t<std::execution::get_stop_token_t, std::in_place_stop_token>> {
          return std::execution::make_env(
            std::execution::get_env(base()),
            std::execution::with(std::execution::get_stop_token, op_state_->stop_source_.get_token()));
        }
        operation_t<CvrefReceiverId>* op_state_;
      };

    template <class CvrefReceiverId>
      struct operation_t {
        using WhenAll = std::__member_t<CvrefReceiverId, when_all_sender_t>;
        using Receiver = std::__t<std::decay_t<CvrefReceiverId>>;
        using Env = std::execution::env_of_t<Receiver>;
        using CvrefEnv = std::__member_t<CvrefReceiverId, Env>;
        using Traits = completion_sigs<CvrefEnv>;

        template <class Sender, std::size_t Index>
          using child_op_state =
            std::execution::connect_result_t<
              std::__member_t<WhenAll, Sender>,
              receiver_t<CvrefReceiverId, Index>>;

        using Indices = std::index_sequence_for<SenderIds...>;

        template <size_t... Is>
          static auto connect_children_(std::index_sequence<Is...>)
            -> std::tuple<child_op_state<std::__t<SenderIds>, Is>...>;

        using child_op_states_tuple_t =
            decltype((connect_children_)(Indices{}));

        void arrive() noexcept {
          if (0 == --count_) {
            complete();
          }
        }

        template <class OpT>
        static void sync(OpT& op) noexcept {
          if constexpr (std::is_base_of_v<detail::op_state_base_t, OpT>) {
            if (op.stream_) {
              THROW_ON_CUDA_ERROR(cudaStreamSynchronize(op.stream_));
            }
          }
        }

        template <std::size_t Index, class... As>
          void store_values(As&&... as) noexcept {
            using SenderId = example::cuda::detail::nth_type<Index, SenderIds...>;
            cudaStream_t stream = std::get<Index>(child_states_).stream_;
            detail::h2d::propagate<true /* async */, SenderId>(stream, [&](auto&&... args) {
              std::get<Index>(this->values_).emplace((As&&)args...);
            }, (As&&)as...);
          }

        void complete() noexcept {
          // Stop callback is no longer needed. Destroy it.
          on_stop_.reset();

          // Synchronize streams
          std::apply([](auto&... ops) { (sync(ops), ...); }, child_states_);

          // All child operations have completed and arrived at the barrier.
          switch(state_.load(std::memory_order_relaxed)) {
          case when_all::started:
            if constexpr (sends_values<Traits>::value) {
              // All child operations completed successfully:
              std::apply(
                [this](auto&... opt_vals) -> void {
                  std::apply(
                    [this](auto&... all_vals) -> void {
                      try {
                      std::execution::set_value(
                            (Receiver&&) recvr_, std::move(all_vals)...);
                      } catch(...) {
                      std::execution::set_error(
                            (Receiver&&) recvr_, std::current_exception());
                      }
                    },
                    std::tuple_cat(
                      std::apply(
                        [](auto&... vals) { return std::tie(vals...); },
                        *opt_vals
                      )...
                    )
                  );
                },
                values_
              );
            }
            break;
          case when_all::error:
            std::visit([this](auto& err) noexcept {
                std::execution::set_error((Receiver&&) recvr_, std::move(err));
            }, errors_);
            break;
          case when_all::stopped:
            std::execution::set_stopped((Receiver&&) recvr_);
            break;
          default:
            ;
          }
        }

        template <size_t... Is>
          operation_t(WhenAll&& when_all, Receiver rcvr, std::index_sequence<Is...>)
            : child_states_{
              std::execution::__conv{[&when_all, this]() {
                  return std::execution::connect(
                      std::get<Is>(((WhenAll&&) when_all).sndrs_),
                      receiver_t<CvrefReceiverId, Is>{{}, {}, this});
                }}...
              }
            , recvr_((Receiver&&) rcvr)
          {}
        operation_t(WhenAll&& when_all, Receiver rcvr)
          : operation_t((WhenAll&&) when_all, (Receiver&&) rcvr, Indices{})
        {}
        _P2300_IMMOVABLE(operation_t);

        friend void tag_invoke(std::execution::start_t, operation_t& self) noexcept {
          // register stop callback:
          self.on_stop_.emplace(
              std::execution::get_stop_token(std::execution::get_env(self.recvr_)),
              when_all::on_stop_requested{self.stop_source_});
          if (self.stop_source_.stop_requested()) {
            // Stop has already been requested. Don't bother starting
            // the child operations.
            std::execution::set_stopped((Receiver&&) self.recvr_);
          } else {
            std::apply([](auto&&... __child_ops) noexcept -> void {
              (std::execution::start(__child_ops), ...);
            }, self.child_states_);
          }
        }

        // tuple<optional<tuple<Vs1...>>, optional<tuple<Vs2...>>, ...>
        using child_values_tuple_t =
          std::__if<
            sends_values<Traits>,
            std::__minvoke<
              std::__q<std::tuple>,
              std::execution::__value_types_of_t<
                std::__t<SenderIds>,
                when_all::env_t<Env>,
                std::__mcompose<std::__q1<std::optional>, std::__q<std::execution::__decayed_tuple>>,
                std::__single_or<void>>...>,
            std::__>;

        child_op_states_tuple_t child_states_;
        Receiver recvr_;
        std::atomic<std::size_t> count_{sizeof...(SenderIds)};
        // Could be non-atomic here and atomic_ref everywhere except __completion_fn
        std::atomic<when_all::state_t> state_{when_all::started};
        std::execution::error_types_of_t<when_all_sender_t, when_all::env_t<Env>, std::execution::__variant> errors_{};
        child_values_tuple_t values_{};
        std::in_place_stop_source stop_source_{};
        std::optional<typename std::execution::stop_token_of_t<std::execution::env_of_t<Receiver>&>::template
            callback_type<when_all::on_stop_requested>> on_stop_{};
      };

    template <std::__decays_to<when_all_sender_t> Self, std::execution::receiver Receiver>
      friend auto tag_invoke(std::execution::connect_t, Self&& self, Receiver&& rcvr)
        -> operation_t<std::__member_t<Self, std::__x<std::decay_t<Receiver>>>> {
        return {(Self&&) self, (Receiver&&) rcvr};
      }

    template <std::__decays_to<when_all_sender_t> Self, class Env>
      friend auto tag_invoke(std::execution::get_completion_signatures_t, Self&&, Env)
        -> completion_sigs<std::__member_t<Self, Env>>;

    std::tuple<std::__t<SenderIds>...> sndrs_;
  };
}

