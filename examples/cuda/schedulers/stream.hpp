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
            kernel<Fun, std::decay_t<As>...><<<1, 1>>>(f_, as...);

            if constexpr (!std::is_base_of_v<receiver_base_t, Receiver>) {
              cudaStreamSynchronize(0);
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

          // TODO GPU storage
          constexpr int block_threads = 256;
          const int grid_blocks = (static_cast<int>(shape_) + block_threads - 1) / block_threads;
          kernel<block_threads, Shape, Fun, std::decay_t<As>...><<<grid_blocks, block_threads>>>(shape_, f_, as...);

          if constexpr (!std::is_base_of_v<receiver_base_t, Receiver>) {
            cudaStreamSynchronize(0);
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

