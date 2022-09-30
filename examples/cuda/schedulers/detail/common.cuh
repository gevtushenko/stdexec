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
#include <cuda/atomic>
#include <cuda/std/tuple>

#include "throw_on_cuda_error.cuh"
#include "queue.cuh"
#include "variant.cuh"
#include "tuple.cuh"

namespace example::cuda {

  enum class device_type {
    host,
    device
  };

#if defined(__clang__) && defined(__CUDA__)
  __host__ inline device_type get_device_type() { return device_type::host; }
  __device__ inline device_type get_device_type() { return device_type::device; }
#else
  __host__ __device__ inline device_type get_device_type() {
    NV_IF_TARGET(NV_IS_HOST,
                 (return device_type::host;),
                 (return device_type::device;));
  }
#endif

  inline __host__ __device__ bool is_on_gpu() {
    return get_device_type() == device_type::device;
  }
}

namespace example::cuda::stream {

  struct context_t;
  struct scheduler_t;
  struct sender_base_t {};
  struct gpu_sender_base_t : sender_base_t {};
  struct receiver_base_t {};

  template <class... Ts>
    using decayed_tuple = tuple_t<std::decay_t<Ts>...>;

  template <class S>
    concept stream_sender =
      std::execution::sender<S> &&
      std::is_base_of_v<sender_base_t, std::decay_t<S>>;

  template <class S>
    concept gpu_stream_sender =
      std::execution::sender<S> &&
      std::is_base_of_v<gpu_sender_base_t, std::decay_t<S>>;

  template <class R>
    concept stream_receiver =
      std::execution::receiver<R> &&
      std::is_base_of_v<receiver_base_t, std::decay_t<R>>;

  namespace detail {
    struct op_state_base_t{};

    namespace h2d {
      template <bool Async, std::size_t I, class T>
        void fetch(cudaStream_t stream, T&) {
          if constexpr (!Async) {
            THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
          }
        }

      template <bool Async, std::size_t I, class T, class Head, class... As>
        void fetch(cudaStream_t stream, T& tpl, Head&& head, As&&... as) {
          THROW_ON_CUDA_ERROR(cudaMemcpyAsync(&std::get<I>(tpl),
                                              &head,
                                              sizeof(std::decay_t<Head>),
                                              cudaMemcpyDeviceToHost,
                                              stream));
          fetch<Async, I + 1>(stream, tpl, (As&&)as...);
        }

      template <bool Async, class SenderId, class Fn, class... As>
        void propagate(cudaStream_t stream, Fn fn, As&&... as) {
          using Sender = std::__t<SenderId>;

          if constexpr (gpu_stream_sender<Sender>) {
            std::tuple<std::decay_t<As>...> h_as;
            fetch<Async, 0>(stream, h_as, (As&&)as...);
            std::apply([&](auto&&... tas) { fn(tas...); }, h_as);
          } else {
            if constexpr (!Async) {
              THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
            }
            fn((As&&)as...);
          }
        }
    }

    template <class EnvId, class VariantId>
      class enqueue_receiver_t : receiver_base_t {
        using Env = std::__t<EnvId>;
        using Variant = std::__t<VariantId>;

        Env env_;
        Variant* variant_;
        queue::task_base_t* task_;
        queue::producer_t producer_;

      public:
        template <std::__one_of<std::execution::set_value_t,
                                std::execution::set_error_t,
                                std::execution::set_stopped_t> Tag,
                  class... As _NVCXX_CAPTURE_PACK(As)>
          friend void tag_invoke(Tag tag, enqueue_receiver_t&& self, As&&... as) noexcept {
            _NVCXX_EXPAND_PACK(As, as,
              self.variant_->template emplace<decayed_tuple<Tag, As...>>(Tag{}, (As&&)as...);
            );
            self.producer_(self.task_);
          }

        /*
        friend void tag_invoke(std::execution::set_error_t, enqueue_receiver_t&& self, const std::exception_ptr&) noexcept {
          self.variant_->template emplace<decayed_tuple<std::execution::set_error_t, cudaError_t>>(
              std::execution::set_error, cudaErrorUnknown);
          self.producer_(self.task_);
        }
        */

        friend Env tag_invoke(std::execution::get_env_t, const enqueue_receiver_t& self) {
          return self.env_;
        }

        enqueue_receiver_t(
            Env env,
            Variant* variant,
            queue::task_base_t* task,
            queue::producer_t producer)
          : env_(env)
          , variant_(variant)
          , task_(task)
          , producer_(producer) {}
      };

    template <class Receiver, class Tag, class... As>
      __launch_bounds__(1) __global__ void continuation_kernel(Receiver receiver, Tag tag, As... as) {
        tag(std::move(receiver), (As&&)as...);
      }

    template <stream_receiver Receiver, class Variant>
      struct continuation_task_t : queue::task_base_t {
        Receiver receiver_;
        Variant* variant_;

        continuation_task_t (Receiver receiver, Variant* variant)
          : receiver_{receiver}
          , variant_{variant} {
          this->execute_ = [](task_base_t* t) noexcept {
            continuation_task_t &self = *reinterpret_cast<continuation_task_t*>(t);

            visit([&self](auto& tpl) {
                apply([&self](auto tag, auto... as) {
                  tag(std::move(self.receiver_), decltype(as)(as)...);
                }, tpl);
            }, *self.variant_);
          };
          this->next_ = nullptr;

          THROW_ON_CUDA_ERROR(cudaMalloc(&this->atom_next_, sizeof(task_base_t*)));
          THROW_ON_CUDA_ERROR(cudaMemset(this->atom_next_, 0, sizeof(task_base_t*)));
        }

        ~continuation_task_t() {
          THROW_ON_CUDA_ERROR(cudaFree(this->atom_next_));
        }
      };
  }

  template <class OuterReceiverId>
    struct operation_state_base_t : detail::op_state_base_t {
      using outer_receiver_t = std::__t<OuterReceiverId>;

      bool owner_{false};
      cudaStream_t stream_{0};
      outer_receiver_t receiver_;

      operation_state_base_t(outer_receiver_t receiver)
        : receiver_(receiver) {}

      template <class Tag, class... As _NVCXX_CAPTURE_PACK(As)>
      void propagate_completion_signal(Tag tag, As&&... as) noexcept {
        _NVCXX_EXPAND_PACK(As, as,
          if constexpr (stream_receiver<outer_receiver_t>) {
            tag((outer_receiver_t&&)receiver_, (As&&)as...);
          } else {
            detail::continuation_kernel<std::decay_t<outer_receiver_t>, Tag, As...><<<1, 1>>>(receiver_, tag, (As&&)as...);
          }
        );
      }

      cudaStream_t allocate() {
        if (stream_ == 0) {
          owner_ = true;
          THROW_ON_CUDA_ERROR(cudaStreamCreate(&stream_));
        }

        return stream_;
      }

      ~operation_state_base_t() {
        if (owner_) {
          THROW_ON_CUDA_ERROR(cudaStreamDestroy(stream_));
          stream_ = 0;
          owner_ = false;
        }
      }
    };

  template <class ReceiverId>
    struct propagate_receiver_t : receiver_base_t {
      operation_state_base_t<ReceiverId>& operation_state_;

      template <std::__one_of<std::execution::set_value_t,
                              std::execution::set_error_t,
                              std::execution::set_stopped_t> Tag,
                class... As  _NVCXX_CAPTURE_PACK(As)>
      friend void tag_invoke(Tag tag, propagate_receiver_t&& self, As&&... as) noexcept {
        _NVCXX_EXPAND_PACK(As, as,
          self.operation_state_.template propagate_completion_signal<Tag, As...>(tag, (As&&)as...);
        );
      }

      friend std::execution::env_of_t<std::__t<ReceiverId>>
      tag_invoke(std::execution::get_env_t, const propagate_receiver_t& self) {
        return std::execution::get_env(self.operation_state_.receiver_);
      }
    };

  namespace detail {
    template <class SenderId, class InnerReceiverId, class OuterReceiverId>
      struct operation_state_t : operation_state_base_t<OuterReceiverId> {
        using sender_t = std::__t<SenderId>;
        using inner_receiver_t = std::__t<InnerReceiverId>;
        using outer_receiver_t = std::__t<OuterReceiverId>;
        using env_t = std::execution::env_of_t<outer_receiver_t>;

        template <class... _Ts>
          using variant =
            std::__minvoke<
              std::__if_c<
                sizeof...(_Ts) != 0,
                std::__transform<std::__q1<std::decay_t>, std::__munique<std::__q<variant_t>>>,
                std::__mconst<std::execution::__not_a_variant>>,
              _Ts...>;

        template <class... _Ts>
          using bind_tuples =
            std::__mbind_front_q<
              variant,
              // tuple_t<std::execution::set_stopped_t>,
              // tuple_t<std::execution::set_error_t, cudaError_t>,
              _Ts...>;

        using bound_values_t =
          std::execution::__value_types_of_t<
            sender_t,
            env_t,
            std::__mbind_front_q<decayed_tuple, std::execution::set_value_t>,
            std::__q<bind_tuples>>;

        using variant_t =
          std::execution::__error_types_of_t<
            sender_t,
            env_t,
            std::__transform<
              // std::__mbind_front<std::__replace<std::exception_ptr, cudaError_t, std::__q<decayed_tuple>>, std::execution::set_error_t>,
              std::__mbind_front_q<decayed_tuple, std::execution::set_error_t>,
              bound_values_t>>;

        using task_t = detail::continuation_task_t<inner_receiver_t, variant_t>;
        using intermediate_receiver = std::__t<std::conditional_t<
          stream_sender<sender_t>,
          std::__x<inner_receiver_t>,
          std::__x<detail::enqueue_receiver_t<std::__x<env_t>, std::__x<variant_t>>>>>;
        using inner_op_state_t = std::execution::connect_result_t<sender_t, intermediate_receiver>;

        queue::task_hub_t* hub_;
        queue::host_ptr<variant_t> storage_;
        queue::host_ptr<task_t> task_;
        inner_op_state_t inner_op_;

        friend void tag_invoke(std::execution::start_t, operation_state_t& op) noexcept {
          op.stream_ = op.get_stream();
          std::execution::start(op.inner_op_);
        }

        cudaStream_t get_stream() {
          cudaStream_t stream{};

          if constexpr (std::is_base_of_v<detail::op_state_base_t, inner_op_state_t>) {
            stream = inner_op_.get_stream();
          } else {
            stream = this->allocate();
          }

          return stream;
        }

        template <std::__decays_to<outer_receiver_t> OutR, class ReceiverProvider>
          requires stream_sender<sender_t>
        operation_state_t(sender_t&& sender, queue::task_hub_t*, OutR&& out_receiver, ReceiverProvider receiver_provider)
          : operation_state_base_t<OuterReceiverId>((outer_receiver_t&&)out_receiver)
          , inner_op_{std::execution::connect((sender_t&&)sender, receiver_provider(*this))}
        {}

        template <std::__decays_to<outer_receiver_t> OutR, class ReceiverProvider>
          requires (!stream_sender<sender_t>)
        operation_state_t(sender_t&& sender, queue::task_hub_t* hub, OutR&& out_receiver, ReceiverProvider receiver_provider)
          : operation_state_base_t<OuterReceiverId>((outer_receiver_t&&)out_receiver)
          , hub_(hub)
          , storage_(queue::make_host<variant_t>())
          , task_(queue::make_host<task_t>(receiver_provider(*this), storage_.get()))
          , inner_op_{
              std::execution::connect((sender_t&&)sender,
              detail::enqueue_receiver_t<std::__x<env_t>, std::__x<variant_t>>{
                std::execution::get_env(out_receiver), storage_.get(), task_.get(), hub_->producer()})}
        {}
      };
  }

  template <class S>
    concept stream_completing_sender =
      std::execution::sender<S> &&
      std::is_same_v<
          std::tag_invoke_result_t<
            std::execution::get_completion_scheduler_t<std::execution::set_value_t>, S>,
          scheduler_t>;

  template <class Sender, class InnerReceiver, class OuterReceiver>
    using stream_op_state_t = detail::operation_state_t<std::__x<Sender>,
                                                        std::__x<InnerReceiver>,
                                                        std::__x<OuterReceiver>>;

  template <stream_completing_sender Sender, class OuterReceiver, class ReceiverProvider>
    stream_op_state_t<Sender, std::invoke_result_t<ReceiverProvider, operation_state_base_t<std::__x<OuterReceiver>>&>, OuterReceiver>
    stream_op_state(Sender&& sndr, OuterReceiver&& out_receiver, ReceiverProvider receiver_provider) {
      detail::queue::task_hub_t* hub = std::execution::get_completion_scheduler<std::execution::set_value_t>(sndr).hub_;

      return stream_op_state_t<
        Sender,
        std::invoke_result_t<ReceiverProvider, operation_state_base_t<std::__x<OuterReceiver>>&>,
        OuterReceiver>(
          (Sender&&)sndr,
          hub,
          (OuterReceiver&&)out_receiver, receiver_provider);
    }

  template <class Sender, class OuterReceiver, class ReceiverProvider>
    stream_op_state_t<Sender, std::invoke_result_t<ReceiverProvider, operation_state_base_t<std::__x<OuterReceiver>>&>, OuterReceiver>
    stream_op_state(detail::queue::task_hub_t* hub, Sender&& sndr, OuterReceiver&& out_receiver, ReceiverProvider receiver_provider) {
      return stream_op_state_t<
        Sender,
        std::invoke_result_t<ReceiverProvider, operation_state_base_t<std::__x<OuterReceiver>>&>,
        OuterReceiver>(
          (Sender&&)sndr,
          hub,
          (OuterReceiver&&)out_receiver, receiver_provider);
    }
}

