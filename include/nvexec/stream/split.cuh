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
#pragma once

#include <stdexec/execution.hpp>
#include <exec/env.hpp>
#include <type_traits>

#include "nvexec/stream/common.cuh"
#include "nvexec/detail/throw_on_cuda_error.cuh"

namespace nvexec::detail::stream {
  namespace split {
    template <class Tag, class Variant, class... As>
      __launch_bounds__(1)
      __global__ void copy_kernel(Variant* var, As&&... as) {
        using tuple_t = decayed_tuple<Tag, As...>;
        var->template emplace<tuple_t>(Tag{}, (As&&)as...);
      }

    template <class SenderId, class SharedState>
      class receiver_t : public stream_receiver_base {
        using Sender = stdexec::__t<SenderId>;

        SharedState &sh_state_;

      public:
        template <stdexec::__one_of<std::execution::set_value_t, 
                                    std::execution::set_error_t, 
                                    std::execution::set_stopped_t> Tag, 
                  class... As _NVCXX_CAPTURE_PACK(As)>
          friend void tag_invoke(Tag tag, receiver_t&& self, As&&... as) noexcept {
            SharedState &state = self.sh_state_;
            cudaStream_t stream = state.op_state2_.stream_;

            if (self.sh_state_.status_ = STDEXEC_DBG_ERR(cudaStreamSynchronize(stream)); 
                self.sh_state_.status_ == cudaSuccess) {
              _NVCXX_EXPAND_PACK(As, as,
                using tuple_t = decayed_tuple<Tag, As...>;
                state.data_->template emplace<tuple_t>(Tag{}, as...);
              )
            } else {
              using tuple_t = decayed_tuple<std::execution::set_error_t, cudaError_t>;
              state.data_->template emplace<tuple_t>(std::execution::set_error, self.sh_state_.status_);
            }
            state.notify();
          }

        friend auto tag_invoke(std::execution::get_env_t, const receiver_t& self)
          -> exec::make_env_t<exec::with_t<std::execution::get_stop_token_t, std::in_place_stop_token>> {
          return exec::make_env(stdexec::__with(std::execution::get_stop_token, self.sh_state_.stop_source_.get_token()));
        }

        explicit receiver_t(SharedState &sh_state_t) noexcept
          : sh_state_(sh_state_t) {
        }
    };

    struct operation_base_t {
      using notify_fn = void(operation_base_t*) noexcept;

      operation_base_t * next_{};
      notify_fn* notify_{};
    };

    template <class SenderId>
      struct sh_state_t {
        using Sender = stdexec::__t<SenderId>;

        template <class... _Ts>
          using bind_tuples =
            stdexec::__mbind_front_q<
              variant_t,
              tuple_t<std::execution::set_stopped_t>, // Initial state of the variant is set_stopped
              tuple_t<std::execution::set_error_t, cudaError_t>,
              _Ts...>;

        using bound_values_t =
          stdexec::__value_types_of_t<
            Sender,
            exec::make_env_t<exec::with_t<std::execution::get_stop_token_t, std::in_place_stop_token>>,
            stdexec::__mbind_front_q<decayed_tuple, std::execution::set_value_t>,
            stdexec::__q<bind_tuples>>;

        using variant_t =
          stdexec::__error_types_of_t<
            Sender,
            exec::make_env_t<exec::with_t<std::execution::get_stop_token_t, std::in_place_stop_token>>,
            stdexec::__transform<
              stdexec::__mbind_front_q<decayed_tuple, std::execution::set_error_t>,
              bound_values_t>>;

        using receiver_ = receiver_t<SenderId, sh_state_t>;
        using inner_op_state_t = std::execution::connect_result_t<Sender, receiver_>;

        std::in_place_stop_source stop_source_{};
        inner_op_state_t op_state2_;
        std::atomic<void*> head_;
        variant_t *data_{nullptr};
        cudaError_t status_{cudaSuccess};

        explicit sh_state_t(Sender& sndr)
          : op_state2_(std::execution::connect((Sender&&) sndr, receiver_{*this}))
          , head_{nullptr} {
          status_ = STDEXEC_DBG_ERR(cudaMallocManaged(&data_, sizeof(variant_t)));
          if (status_ == cudaSuccess) {
            new (data_) variant_t();
          }
        }

        ~sh_state_t() {
          if (data_) {
            STDEXEC_DBG_ERR(cudaFree(data_));
          }
        }

        void notify() noexcept {
          void* const completion_state = static_cast<void*>(this);
          void *old = head_.exchange(completion_state, std::memory_order_acq_rel);
          operation_base_t *op_state = static_cast<operation_base_t*>(old);

          while(op_state != nullptr) {
            operation_base_t *next = op_state->next_;
            op_state->notify_(op_state);
            op_state = next;
          }
        }
      };

    template <class SenderId, class ReceiverId>
      class operation_t : public operation_base_t
                        , public operation_state_base_t<ReceiverId> {
        using Sender = stdexec::__t<SenderId>;
        using Receiver = stdexec::__t<ReceiverId>;

        struct on_stop_requested {
          std::in_place_stop_source& stop_source_;
          void operator()() noexcept {
            stop_source_.request_stop();
          }
        };
        using on_stop = std::optional<typename std::execution::stop_token_of_t<
            std::execution::env_of_t<Receiver> &>::template callback_type<on_stop_requested>>;

        on_stop on_stop_{};
        std::shared_ptr<sh_state_t<SenderId>> shared_state_;

      public:
        operation_t(Receiver&& rcvr,
                    std::shared_ptr<sh_state_t<SenderId>> shared_state)
            noexcept(std::is_nothrow_move_constructible_v<Receiver>)
          : operation_base_t{nullptr, notify}
          , operation_state_base_t<ReceiverId>((Receiver&&)rcvr)
          , shared_state_(move(shared_state)) {
        }
        STDEXEC_IMMOVABLE(operation_t);

        static void notify(operation_base_t* self) noexcept {
          operation_t *op = static_cast<operation_t*>(self);
          op->on_stop_.reset();

          cudaError_t status = op->shared_state_->status_;
          if (status == cudaSuccess) {
            visit([&](auto& tupl) noexcept -> void {
              apply([&](auto tag, auto&... args) noexcept -> void {
                op->propagate_completion_signal(tag, args...);
              }, tupl);
            }, *op->shared_state_->data_);
          } else {
            op->propagate_completion_signal(std::execution::set_error, std::move(status));
          }
        }

        cudaStream_t get_stream() {
          return this->allocate();
        }

        friend void tag_invoke(std::execution::start_t, operation_t& self) noexcept {
          sh_state_t<SenderId>* shared_state = self.shared_state_.get();
          std::atomic<void*>& head = shared_state->head_;
          void* const completion_state = static_cast<void*>(shared_state);
          void* old = head.load(std::memory_order_acquire);

          if (old != completion_state) {
            self.on_stop_.emplace(
                std::execution::get_stop_token(std::execution::get_env(self.receiver_)),
                on_stop_requested{shared_state->stop_source_});
          }

          do {
            if (old == completion_state) {
              self.notify(&self);
              return;
            }
            self.next_ = static_cast<operation_base_t*>(old);
          } while (!head.compare_exchange_weak(
              old, static_cast<void *>(&self),
              std::memory_order_release,
              std::memory_order_acquire));

          if (old == nullptr) {
            // the inner sender isn't running
            if (shared_state->stop_source_.stop_requested()) {
              // 1. resets head to completion state
              // 2. notifies waiting threads
              // 3. propagates "stopped" signal to `out_r'`
              shared_state->notify();
            } else {
              std::execution::start(shared_state->op_state2_);
            }
          }
        }
      };
  } // namespace split

  template <class SenderId>
    class split_sender_t : stream_sender_base {
      using Sender = stdexec::__t<SenderId>;
      using sh_state_ = split::sh_state_t<SenderId>;
      template <class Receiver>
        using operation_t = split::operation_t<SenderId, stdexec::__x<std::remove_cvref_t<Receiver>>>;

      Sender sndr_;
      std::shared_ptr<sh_state_> shared_state_;

    public:
      template <stdexec::__decays_to<split_sender_t> Self, std::execution::receiver Receiver>
          requires std::execution::receiver_of<Receiver, std::execution::completion_signatures_of_t<Self, stdexec::__empty_env>>
        friend auto tag_invoke(std::execution::connect_t, Self&& self, Receiver&& recvr)
          noexcept(std::is_nothrow_constructible_v<std::decay_t<Receiver>, Receiver>)
          -> operation_t<Receiver> {
          return operation_t<Receiver>{(Receiver &&) recvr,
                                        self.shared_state_};
        }

      template <stdexec::tag_category<std::execution::forwarding_sender_query> Tag, class... As _NVCXX_CAPTURE_PACK(As)>
          requires // Always complete on GPU, so no need in (!stdexec::__is_instance_of<Tag, std::execution::get_completion_scheduler_t>) && 
            stdexec::__callable<Tag, const Sender&, As...>
        friend auto tag_invoke(Tag tag, const split_sender_t& self, As&&... as)
          noexcept(stdexec::__nothrow_callable<Tag, const Sender&, As...>)
          -> stdexec::__call_result_if_t<stdexec::tag_category<Tag, std::execution::forwarding_sender_query>, Tag, const Sender&, As...> {
          _NVCXX_EXPAND_PACK_RETURN(As, as,
            return ((Tag&&) tag)(self.sndr_, (As&&) as...);
          )
        }

      template <class... Tys>
        using set_value_t = std::execution::completion_signatures<std::execution::set_value_t(const std::decay_t<Tys>&...)>;

      template <class Ty>
        using set_error_t = std::execution::completion_signatures<std::execution::set_error_t(const std::decay_t<Ty>&)>;

      template <stdexec::__decays_to<split_sender_t> Self, class Env>
        friend auto tag_invoke(std::execution::get_completion_signatures_t, Self&&, Env) ->
          std::execution::make_completion_signatures<
            Sender,
            exec::make_env_t<exec::with_t<std::execution::get_stop_token_t, std::in_place_stop_token>>,
            std::execution::completion_signatures<std::execution::set_error_t(cudaError_t)>,
            set_value_t,
            set_error_t>;

      explicit split_sender_t(Sender sndr)
          : sndr_((Sender&&) sndr)
          , shared_state_{std::make_shared<sh_state_>(sndr_)}
      {}
    };
}

