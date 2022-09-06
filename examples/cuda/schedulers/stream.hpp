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

#include <thrust/device_vector.h>


namespace example::cuda::stream {
  struct receiver_base_t {};
  struct operation_state_base_t {
    cudaStream_t stream_{0};
  };
}


namespace _P2300::execution {
  using namespace _P2300;
  using std::remove_cvref_t;

  namespace stream_when_all {
    namespace __impl {
      enum __state_t { __started, __error, __stopped };

      struct __on_stop_requested {
        in_place_stop_source& __stop_source_;
        void operator()() noexcept {
          __stop_source_.request_stop();
        }
      };

      template <class _Env>
        using __env_t =
          make_env_t<_Env, with_t<get_stop_token_t, in_place_stop_token>>;

      template <class...>
        using __swallow_values = completion_signatures<>;

      template <class _Env, class... _Senders>
        struct __traits {
          using __t = dependent_completion_signatures<_Env>;
        };

      template <class _Env, class... _Senders>
          requires ((__v<__count_of<set_value_t, _Senders, _Env>> <= 1) &&...)
        struct __traits<_Env, _Senders...> {
          using __non_values =
            __concat_completion_signatures_t<
              completion_signatures<
                set_error_t(std::exception_ptr),
                set_stopped_t()>,
              make_completion_signatures<
                _Senders,
                _Env,
                completion_signatures<>,
                __swallow_values>...>;
          using __values =
            __minvoke<
              __concat<__qf<set_value_t>>,
              __value_types_of_t<
                _Senders,
                _Env,
                __q<__types>,
                __single_or<__types<>>>...>;
          using __t =
            __if_c<
              (__sends<set_value_t, _Senders, _Env> &&...),
              __minvoke2<
                __push_back<__q<completion_signatures>>, __non_values, __values>,
              __non_values>;
        };

      template <class... _SenderIds>
        struct __sender {
          template <class... _Sndrs>
            explicit __sender(_Sndrs&&... __sndrs)
              : __sndrs_((_Sndrs&&) __sndrs...)
            {}

         private:
          template <class _CvrefEnv>
            using __completion_sigs =
              __t<__traits<
                __env_t<remove_cvref_t<_CvrefEnv>>,
                __member_t<_CvrefEnv, __t<_SenderIds>>...>>;

          template <class _Traits>
            using __sends_values =
              __bool<__v<typename _Traits::template
                __gather_sigs<set_value_t, __mconst<int>, __mcount>> != 0>;

          template <class _CvrefReceiverId>
            struct __operation;

          template <class _CvrefReceiverId, std::size_t _Index>
            struct __receiver : receiver_adaptor<__receiver<_CvrefReceiverId, _Index>>, example::cuda::stream::receiver_base_t {
              using _WhenAll = __member_t<_CvrefReceiverId, __sender>;
              using _Receiver = __t<decay_t<_CvrefReceiverId>>;
              using _Traits =
                __completion_sigs<
                  __member_t<_CvrefReceiverId, env_of_t<_Receiver>>>;

              _Receiver&& base() && noexcept {
                return (_Receiver&&) __op_state_->__recvr_;
              }
              const _Receiver& base() const & noexcept {
                return __op_state_->__recvr_;
              }
              template <class _Error>
                void __set_error(_Error&& __err, __state_t __expected) noexcept {
                  // TODO: _What memory orderings are actually needed here?
                  if (__op_state_->__state_.compare_exchange_strong(__expected, __error)) {
                    __op_state_->__stop_source_.request_stop();
                    // We won the race, free to write the error into the operation
                    // state without worry.
                    try {
                      __op_state_->__errors_.template emplace<decay_t<_Error>>((_Error&&) __err);
                    } catch(...) {
                      __op_state_->__errors_.template emplace<std::exception_ptr>(std::current_exception());
                    }
                  }
                  __op_state_->__arrive();
                }
              template <class... _Values>
                void set_value(_Values&&... __vals) && noexcept {
                  if constexpr (__sends_values<_Traits>::value) {
                    // We only need to bother recording the completion values
                    // if we're not already in the "error" or "stopped" state.
                    if (__op_state_->__state_ == __started) {
                      try {
                        std::get<_Index>(__op_state_->__values_).emplace(
                            (_Values&&) __vals...);
                      } catch(...) {
                        __set_error(std::current_exception(), __started);
                      }
                    }
                  }
                  __op_state_->__arrive();
                }
              template <class _Error>
                  requires tag_invocable<set_error_t, _Receiver, _Error>
                void set_error(_Error&& __err) && noexcept {
                  __set_error((_Error&&) __err, __started);
                }
              void set_stopped() && noexcept {
                __state_t __expected = __started;
                // Transition to the "stopped" state if and only if we're in the
                // "started" state. (If this fails, it's because we're in an
                // error state, which trumps cancellation.)
                if (__op_state_->__state_.compare_exchange_strong(__expected, __stopped)) {
                  __op_state_->__stop_source_.request_stop();
                }
                __op_state_->__arrive();
              }
              auto get_env() const
                -> make_env_t<env_of_t<_Receiver>, with_t<get_stop_token_t, in_place_stop_token>> {
                return make_env(
                  execution::get_env(base()),
                  with(get_stop_token, __op_state_->__stop_source_.get_token()));
              }
              __operation<_CvrefReceiverId>* __op_state_;
            };

          template <class _CvrefReceiverId>
            struct __operation {
              using _WhenAll = __member_t<_CvrefReceiverId, __sender>;
              using _Receiver = __t<decay_t<_CvrefReceiverId>>;
              using _Env = env_of_t<_Receiver>;
              using _CvrefEnv = __member_t<_CvrefReceiverId, _Env>;
              using _Traits = __completion_sigs<_CvrefEnv>;

              template <class _Sender, std::size_t _Index>
                using __child_op_state =
                  connect_result_t<
                    __member_t<_WhenAll, _Sender>,
                    __receiver<_CvrefReceiverId, _Index>>;

              using _Indices = std::index_sequence_for<_SenderIds...>;

              template <size_t... _Is>
                static auto __connect_children_(std::index_sequence<_Is...>)
                  -> std::tuple<__child_op_state<__t<_SenderIds>, _Is>...>;

              using __child_op_states_tuple_t =
                  decltype((__connect_children_)(_Indices{}));

              void __arrive() noexcept {
                if (0 == --__count_) {
                  __complete();
                }
              }

              void __complete() noexcept {
                // Stop callback is no longer needed. Destroy it.
                __on_stop_.reset();
                // All child operations have completed and arrived at the barrier.
                switch(__state_.load(std::memory_order_relaxed)) {
                case __started:
                  if constexpr (__sends_values<_Traits>::value) {
                    cudaDeviceSynchronize();

                    // All child operations completed successfully:
                    std::apply(
                      [this](auto&... __opt_vals) -> void {
                        std::apply(
                          [this](auto&... __all_vals) -> void {
                            try {
                              execution::set_value(
                                  (_Receiver&&) __recvr_, std::move(__all_vals)...);
                            } catch(...) {
                              execution::set_error(
                                  (_Receiver&&) __recvr_, std::current_exception());
                            }
                          },
                          std::tuple_cat(
                            std::apply(
                              [](auto&... __vals) { return std::tie(__vals...); },
                              *__opt_vals
                            )...
                          )
                        );
                      },
                      __values_
                    );
                  }
                  break;
                case __error:
                  std::visit([this](auto& __err) noexcept {
                    execution::set_error((_Receiver&&) __recvr_, std::move(__err));
                  }, __errors_);
                  break;
                case __stopped:
                  execution::set_stopped((_Receiver&&) __recvr_);
                  break;
                default:
                  ;
                }
              }

              template <size_t... _Is>
                __operation(_WhenAll&& __when_all, _Receiver __rcvr, std::index_sequence<_Is...>)
                  : __child_states_{
                      __conv{[&__when_all, this]() {
                        return execution::connect(
                            std::get<_Is>(((_WhenAll&&) __when_all).__sndrs_),
                            __receiver<_CvrefReceiverId, _Is>{{}, {}, this});
                      }}...
                    }
                  , __recvr_((_Receiver&&) __rcvr)
                {}
              __operation(_WhenAll&& __when_all, _Receiver __rcvr)
                : __operation((_WhenAll&&) __when_all, (_Receiver&&) __rcvr, _Indices{})
              {}
              _P2300_IMMOVABLE(__operation);

              friend void tag_invoke(start_t, __operation& __self) noexcept {
                // register stop callback:
                __self.__on_stop_.emplace(
                    get_stop_token(get_env(__self.__recvr_)),
                    __on_stop_requested{__self.__stop_source_});
                if (__self.__stop_source_.stop_requested()) {
                  // Stop has already been requested. Don't bother starting
                  // the child operations.
                  execution::set_stopped((_Receiver&&) __self.__recvr_);
                } else {
                  apply([](auto&&... __child_ops) noexcept -> void {
                    (execution::start(__child_ops), ...);
                  }, __self.__child_states_);
                }
              }

              // tuple<optional<tuple<Vs1...>>, optional<tuple<Vs2...>>, ...>
              using __child_values_tuple_t =
                __if<
                  __sends_values<_Traits>,
                  __minvoke<
                    __q<std::tuple>,
                    __value_types_of_t<
                      __t<_SenderIds>,
                      __env_t<_Env>,
                      __mcompose<__q1<std::optional>, __q<__decayed_tuple>>,
                      __single_or<void>>...>,
                  __>;

              __child_op_states_tuple_t __child_states_;
              _Receiver __recvr_;
              std::atomic<std::size_t> __count_{sizeof...(_SenderIds)};
              // Could be non-atomic here and atomic_ref everywhere except __completion_fn
              std::atomic<__state_t> __state_{__started};
              error_types_of_t<__sender, __env_t<_Env>, __variant> __errors_{};
              [[no_unique_address]] __child_values_tuple_t __values_{};
              in_place_stop_source __stop_source_{};
              std::optional<typename stop_token_of_t<env_of_t<_Receiver>&>::template
                  callback_type<__on_stop_requested>> __on_stop_{};
            };

          template <__decays_to<__sender> _Self, receiver _Receiver>
            friend auto tag_invoke(connect_t, _Self&& __self, _Receiver&& __rcvr)
              -> __operation<__member_t<_Self, __x<decay_t<_Receiver>>>> {
              return {(_Self&&) __self, (_Receiver&&) __rcvr};
            }

          template <__decays_to<__sender> _Self, class _Env>
            friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env)
              -> __completion_sigs<__member_t<_Self, _Env>>;

          std::tuple<__t<_SenderIds>...> __sndrs_;
        };
    } // namespce __impl
  } // namespace stream_when_all
}


namespace example::cuda::stream {

  namespace then {

    template <class Fun, class... As>
      __launch_bounds__(1) 
      __global__ void kernel(Fun fn, As... as) {
        fn(as...);
      }

    template <class Fun, class ResultT, class... As>
      __launch_bounds__(1) 
      __global__ void kernel_with_result(Fun fn, ResultT* result, As... as) {
        *result = fn(as...);
      }

    template <class ReceiverId, class Fun>
      class receiver_t
        : std::execution::receiver_adaptor<receiver_t<ReceiverId, Fun>, std::__t<ReceiverId>>
        , receiver_base_t {
        using Receiver = std::__t<ReceiverId>;
        friend std::execution::receiver_adaptor<receiver_t, Receiver>;

        Fun f_;
        operation_state_base_t &op_state_;

        template <class... As>
        void set_value(As&&... as) && noexcept 
          requires std::__callable<Fun, std::decay_t<As>...> {
          using result_t = std::decay_t<std::invoke_result_t<Fun, std::decay_t<As>...>>;

          if constexpr (std::is_same_v<void, result_t>) {
            kernel<Fun, std::decay_t<As>...><<<1, 1, 0, op_state_.stream_>>>(f_, as...);

            if constexpr (!std::is_base_of_v<receiver_base_t, Receiver>) {
              cudaStreamSynchronize(op_state_.stream_);
            }

            std::execution::set_value(std::move(this->base()));
          } else {
            thrust::device_vector<result_t> result(1);
            kernel_with_result<Fun, std::decay_t<As>...><<<1, 1>>>(f_, thrust::raw_pointer_cast(result.data()), as...);
            cudaStreamSynchronize(0);
            std::execution::set_value(std::move(this->base()), static_cast<result_t>(result[0]));
          }
        }

       public:
        explicit receiver_t(Receiver rcvr, Fun fun, operation_state_base_t &op_state)
          : std::execution::receiver_adaptor<receiver_t, Receiver>((Receiver&&) rcvr)
          , f_((Fun&&) fun)
          , op_state_(op_state)
        {}
      };

    template <class SenderId, class ReceiverId, class Fun>
      struct op_state_t : operation_state_base_t {
        using Sender = std::__t<SenderId>;
        using Receiver = std::__t<ReceiverId>;
        using then_receiver_t = receiver_t<ReceiverId, Fun>;
        using inner_op_state = std::execution::connect_result_t<Sender, then_receiver_t>;

        inner_op_state inner_op_;

        friend void tag_invoke(std::execution::start_t, op_state_t& op) noexcept {
          cudaStreamCreate(&op.stream_);
          std::execution::start(op.inner_op_);
        }

        operation_state_base_t& get_stream_provider() {
          if constexpr (std::is_base_of_v<operation_state_base_t, inner_op_state>) {
            return inner_op_;
          }

          return *this;
        }

        op_state_t(Fun fn, Sender&& sender, Receiver receiver)
          : inner_op_{ 
              std::execution::connect(
                  (Sender&&)sender, 
                  then_receiver_t((Receiver&&)receiver, fn, get_stream_provider())) } 
        { }

        ~op_state_t() {
          if (stream_) {
            cudaStreamDestroy(stream_);
            stream_ = 0;
          }
        }
      };

    template <class SenderId, class FunId>
      struct sender_t {
        using Sender = std::__t<SenderId>;
        using Fun = std::__t<FunId>;

        Sender sndr_;
        Fun fun_;

        using set_error_t = 
          std::execution::completion_signatures<
            std::execution::set_error_t(std::exception_ptr)>;

        template <class Self, class Receiver>
          using operation_state_t = 
            op_state_t<
              std::__x<std::__member_t<Self, Sender>>, 
              std::__x<std::remove_cvref_t<Receiver>>, 
              Fun>;

        template <class Self, class Env>
          using completion_signatures =
            std::execution::__make_completion_signatures<
              std::__member_t<Self, Sender>,
              Env,
              std::execution::__with_error_invoke_t<
                std::execution::set_value_t, 
                Fun, 
                std::__member_t<Self, Sender>, 
                Env>,
              std::__mbind_front_q<std::execution::__set_value_invoke_t, Fun>>;

        template <std::__decays_to<sender_t> Self, std::execution::receiver Receiver>
          requires std::execution::receiver_of<Receiver, completion_signatures<Self, std::execution::env_of_t<Receiver>>>
        friend auto tag_invoke(std::execution::connect_t, Self&& self, Receiver&& rcvr)
          noexcept(std::is_nothrow_constructible_v<operation_state_t<Self, Receiver>, Fun, Sender, Receiver>) 
          -> operation_state_t<Self, Receiver> {
          return operation_state_t<Self, Receiver>(self.fun_, ((Self&&)self).sndr_, (Receiver&&)rcvr);
        }

        template <std::__decays_to<sender_t> Self, class Env>
        friend auto tag_invoke(std::execution::get_completion_signatures_t, Self&&, Env)
          -> std::execution::dependent_completion_signatures<Env>;

        template <std::__decays_to<sender_t> Self, class Env>
        friend auto tag_invoke(std::execution::get_completion_signatures_t, Self&&, Env)
          -> completion_signatures<Self, Env> requires true;

        template <std::execution::tag_category<std::execution::forwarding_sender_query> Tag, class... As>
          requires std::__callable<Tag, const Sender&, As...>
        friend auto tag_invoke(Tag tag, const sender_t& self, As&&... as)
          noexcept(std::__nothrow_callable<Tag, const Sender&, As...>)
          -> std::__call_result_if_t<std::execution::tag_category<Tag, std::execution::forwarding_sender_query>, Tag, const Sender&, As...> {
          return ((Tag&&) tag)(self.sndr_, (As&&) as...);
        }
      };
  }

  namespace bulk {

    template <int BlockThreads, std::integral Shape, class Fun, class... As>
      __launch_bounds__(BlockThreads) 
      __global__ void kernel(Shape shape, Fun fn, As... as) {
        const int tid = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);

        if (tid < static_cast<int>(shape)) {
          fn(tid, as...);
        }
      }

    template <class T>
    void print();

    template <class ReceiverId, std::integral Shape, class Fun>
      class receiver_t
        : std::execution::receiver_adaptor<receiver_t<ReceiverId, Shape, Fun>, std::__t<ReceiverId>>
        , receiver_base_t {
        using Receiver = std::__t<ReceiverId>;
        friend std::execution::receiver_adaptor<receiver_t, Receiver>;

        Shape shape_;
        Fun f_;

        operation_state_base_t& op_state_;

        template <class... As>
        void set_value(As&&... as) && noexcept 
          requires std::__callable<Fun, Shape, std::decay_t<As>...> {

          constexpr int block_threads = 256;
          const int grid_blocks = (static_cast<int>(shape_) + block_threads - 1) / block_threads;
          kernel<block_threads, Shape, Fun, std::decay_t<As>...><<<grid_blocks, block_threads, 0, op_state_.stream_>>>(shape_, f_, as...);

          if constexpr (!std::is_base_of_v<receiver_base_t, Receiver>) {
            cudaStreamSynchronize(op_state_.stream_);
          }

          std::execution::set_value(std::move(this->base()), (As&&)as...);
        }

       public:
        explicit receiver_t(Receiver rcvr, Shape shape, Fun fun, operation_state_base_t& op_state)
          : std::execution::receiver_adaptor<receiver_t, Receiver>((Receiver&&) rcvr)
          , shape_(shape)
          , f_((Fun&&) fun)
          , op_state_(op_state)
        {}
      };

    template <class SenderId, class ReceiverId, std::integral Shape, class Fun>
      struct op_state_t : operation_state_base_t {
        using Sender = std::__t<SenderId>;
        using Receiver = std::__t<ReceiverId>;
        using bulk_receiver_t = receiver_t<ReceiverId, Shape, Fun>;
        using inner_op_state = std::execution::connect_result_t<Sender, bulk_receiver_t>;

        inner_op_state inner_op_;

        friend void tag_invoke(std::execution::start_t, op_state_t& op) noexcept {
          cudaStreamCreate(&op.stream_);
          std::execution::start(op.inner_op_);
        }

        operation_state_base_t& get_stream_provider() {
          if constexpr (std::is_base_of_v<operation_state_base_t, inner_op_state>) {
            return inner_op_;
          }

          return *this;
        }

        op_state_t(Shape shape, Fun fn, Sender&& sender, Receiver receiver)
          : inner_op_{ 
              std::execution::connect(
                  (Sender&&)sender, 
                  bulk_receiver_t((Receiver&&)receiver, shape, fn, get_stream_provider())) } 
        { }

        ~op_state_t() {
          if (stream_) {
            cudaStreamDestroy(stream_);
            stream_ = 0;
          }
        }
      };

    template <class SenderId, std::integral Shape, class FunId>
      struct sender_t {
        using Sender = std::__t<SenderId>;
        using Fun = std::__t<FunId>;

        Sender sndr_;
        Shape shape_;
        Fun fun_;

        using set_error_t = 
          std::execution::completion_signatures<
            std::execution::set_error_t(std::exception_ptr)>;

        template <class Self, class Receiver>
          using operation_state_t = 
            op_state_t<
              std::__x<std::__member_t<Self, Sender>>, 
              std::__x<std::remove_cvref_t<Receiver>>, 
              Shape,
              Fun>;

        template <class... Tys>
        using set_value_t = 
          std::execution::completion_signatures<
            std::execution::set_value_t(std::decay_t<Tys>...)>;

        template <class Self, class Env>
          using completion_signatures =
            std::execution::__make_completion_signatures<
              std::__member_t<Self, Sender>,
              Env,
              set_error_t,
              std::__q<set_value_t>>;

        template <std::__decays_to<sender_t> Self, std::execution::receiver Receiver>
          requires std::execution::receiver_of<Receiver, completion_signatures<Self, std::execution::env_of_t<Receiver>>>
        friend auto tag_invoke(std::execution::connect_t, Self&& self, Receiver&& rcvr)
          noexcept(std::is_nothrow_constructible_v<operation_state_t<Self, Receiver>, Fun, Sender, Receiver>) 
          -> operation_state_t<Self, Receiver> {
          return operation_state_t<Self, Receiver>(self.shape_, self.fun_, ((Self&&)self).sndr_, (Receiver&&)rcvr);
        }

        template <std::__decays_to<sender_t> Self, class Env>
        friend auto tag_invoke(std::execution::get_completion_signatures_t, Self&&, Env)
          -> std::execution::dependent_completion_signatures<Env>;

        template <std::__decays_to<sender_t> Self, class Env>
        friend auto tag_invoke(std::execution::get_completion_signatures_t, Self&&, Env)
          -> completion_signatures<Self, Env> requires true;

        template <std::execution::tag_category<std::execution::forwarding_sender_query> Tag, class... As>
          requires std::__callable<Tag, const Sender&, As...>
        friend auto tag_invoke(Tag tag, const sender_t& self, As&&... as)
          noexcept(std::__nothrow_callable<Tag, const Sender&, As...>)
          -> std::__call_result_if_t<std::execution::tag_category<Tag, std::execution::forwarding_sender_query>, Tag, const Sender&, As...> {
          return ((Tag&&) tag)(self.sndr_, (As&&) as...);
        }
      };
  }

  struct scheduler_t {
    template <class R_>
      struct operation_state_t {
        using R = std::__t<R_>;
        [[no_unique_address]] R rec_;
        friend void tag_invoke(std::execution::start_t, operation_state_t& op) noexcept try {
          std::execution::set_value((R&&) op.rec_);
        } catch(...) {
          std::execution::set_error((R&&) op.rec_, std::current_exception());
        }
      };

    struct sender_t {
      using completion_signatures =
        std::execution::completion_signatures<
          std::execution::set_value_t(),
          std::execution::set_error_t(std::exception_ptr)>;

      template <class R>
        friend auto tag_invoke(std::execution::connect_t, sender_t, R&& rec)
          noexcept(std::is_nothrow_constructible_v<std::remove_cvref_t<R>, R>)
          -> operation_state_t<std::__x<std::remove_cvref_t<R>>> {
          return {(R&&) rec};
        }

      friend scheduler_t
      tag_invoke(std::execution::get_completion_scheduler_t<std::execution::set_value_t>, sender_t) noexcept {
        return {};
      }
    };

    template <std::execution::sender Sender, std::integral Shape, class Fun>
      using bulk_sender_t = bulk::sender_t<std::__x<std::remove_cvref_t<Sender>>, Shape, std::__x<std::remove_cvref_t<Fun>>>;

    template <std::execution::sender Sender, class Fun>
      using then_sender_t = then::sender_t<std::__x<std::remove_cvref_t<Sender>>, std::__x<std::remove_cvref_t<Fun>>>;

    template <std::execution::sender S, std::integral Shape, class Fn>
    friend bulk_sender_t<S, Shape, Fn>
    tag_invoke(std::execution::bulk_t, const scheduler_t& sch, S&& sndr, Shape shape, Fn fun) noexcept {
      return bulk_sender_t<S, Shape, Fn>{(S&&) sndr, shape, (Fn&&)fun};
    }

    template <std::execution::sender S, class Fn>
    friend then_sender_t<S, Fn>
    tag_invoke(std::execution::then_t, const scheduler_t& sch, S&& sndr, Fn fun) noexcept {
      return then_sender_t<S, Fn>{(S&&) sndr, (Fn&&)fun};
    }

    template <std::execution::sender... Senders>
    friend auto // _P2300::execution::stream_when_all::__impl::__sender<std::__x<std::decay_t<Senders>>...>
    tag_invoke(std::execution::transfer_when_all_t, const scheduler_t& sch, Senders&&... sndrs) noexcept {
      return std::execution::transfer(_P2300::execution::stream_when_all::__impl::__sender<std::__x<std::decay_t<Senders>>...>{(Senders&&)sndrs...}, sch);
    }

    friend sender_t tag_invoke(std::execution::schedule_t, const scheduler_t&) noexcept {
      return {};
    }

    friend std::execution::forward_progress_guarantee tag_invoke(
        std::execution::get_forward_progress_guarantee_t,
        const scheduler_t&) noexcept {
      return std::execution::forward_progress_guarantee::weakly_parallel;
    }

    bool operator==(const scheduler_t&) const noexcept = default;
  };
}

