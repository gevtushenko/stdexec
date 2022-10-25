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

#include <atomic>
#include "../../stdexec/execution.hpp"

#include <cuda/std/type_traits>
#include <cuda/std/tuple>
#include <optional>

#include "../detail/config.cuh"
#include "../detail/cuda_atomic.cuh"
#include "../detail/throw_on_cuda_error.cuh"
#include "../detail/queue.cuh"
#include "../detail/variant.cuh"

namespace nvexec {

  enum class stream_priority {
    high, 
    normal, 
    low
  };

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

  inline bool is_on_gpu() {
    return get_device_type() == device_type::device;
  }
}

namespace nvexec {
  struct stream_context;

  namespace STDEXEC_STREAM_DETAIL_NS {
    struct context_state_t {
      queue::task_hub_t* hub_{nullptr};
      std::optional<cudaStream_t> stream_;
      stream_priority priority_;

      context_state_t(
        queue::task_hub_t* hub,
        stream_priority priority = stream_priority::normal,
        std::optional<cudaStream_t> stream = {})
        : hub_(hub)
        , stream_(stream) 
        , priority_(priority) {
      }
    };

    inline std::pair<int, cudaError_t> get_stream_priority(stream_priority priority) {
      int least{};
      int greatest{};

      if (cudaError_t status = STDEXEC_DBG_ERR(cudaDeviceGetStreamPriorityRange(&least, &greatest));
          status != cudaSuccess) {
        return std::make_pair(0, status);
      }

      if (priority == stream_priority::low) {
        return std::make_pair(least, cudaSuccess);
      } else if (priority == stream_priority::high) {
        return std::make_pair(greatest, cudaSuccess);
      }

      return std::make_pair(0, cudaSuccess);
    }

    inline std::pair<cudaStream_t, cudaError_t> 
    create_stream_with_priority(stream_priority priority) {
      cudaStream_t stream{};
      cudaError_t status{cudaSuccess};

      if (priority == stream_priority::normal) {
        status = 
          STDEXEC_DBG_ERR(cudaStreamCreate(&stream));
      } else {
        int cuda_priority{};
        std::tie(cuda_priority, status) = get_stream_priority(priority);

        if (status != cudaSuccess) {
          return std::make_pair(cudaStream_t{}, status);
        }

        status = 
          STDEXEC_DBG_ERR(cudaStreamCreateWithPriority(&stream, cudaStreamDefault, cuda_priority));
      }

      return std::make_pair(stream, status);
    }

    struct stream_scheduler;
    struct stream_sender_base {};
    struct stream_receiver_base { constexpr static std::size_t memory_allocation_size = 0; };
    struct stream_env_base { context_state_t context_state_; };

    struct get_stream_t {
      template <stdexec::__none_of<stdexec::no_env> Env>
        std::optional<cudaStream_t> operator()(const Env&) const noexcept {
          return std::nullopt;
        }
      template <stdexec::__none_of<stdexec::no_env> Env>
          requires (std::is_base_of_v<stream_env_base, Env>)
        std::optional<cudaStream_t> operator()(const Env& env) const noexcept {
          return env.context_state_.stream_;
        }
    };

    template <class... Ts>
      using decayed_tuple = ::cuda::std::tuple<std::decay_t<Ts>...>;

    namespace stream_storage_impl {
      template <class... _Ts>
        using variant =
          stdexec::__minvoke<
            stdexec::__if_c<
              sizeof...(_Ts) != 0,
              stdexec::__transform<stdexec::__q1<std::decay_t>, stdexec::__munique<stdexec::__q<variant_t>>>,
              stdexec::__mconst<stdexec::__not_a_variant>>,
            _Ts...>;

      template <class... _Ts>
        using bind_tuples =
          stdexec::__mbind_front_q<
            variant,
            ::cuda::std::tuple<std::execution::set_stopped_t>,
            ::cuda::std::tuple<std::execution::set_error_t, cudaError_t>,
            _Ts...>;

      template <class Sender, class Env>
        using bound_values_t =
          stdexec::__value_types_of_t<
            Sender,
            Env,
            stdexec::__mbind_front_q<decayed_tuple, std::execution::set_value_t>,
            stdexec::__q<bind_tuples>>;
    }

    template <class Sender, class Env>
      using variant_storage_t =
        stdexec::__error_types_of_t<
          Sender,
          Env,
          stdexec::__transform<
            stdexec::__mbind_front_q<decayed_tuple, std::execution::set_error_t>,
            stream_storage_impl::bound_values_t<Sender, Env>>>;

    inline constexpr get_stream_t get_stream{};

    template <class BaseEnvId>
      struct stream_env : stream_env_base {
        using BaseEnv = stdexec::__t<BaseEnvId>;
        BaseEnv base_env_;

        template <stdexec::__none_of<get_stream_t> Tag, 
                  stdexec::same_as<stream_env> Self,
                  class... As>
            requires stdexec::__callable<Tag, const BaseEnv&, As...>
          friend auto tag(Tag tag, const Self& self, As&&... as) noexcept {
            return ((Tag&&)tag)(self.base_env_, (As&&)as...);
          }
      };

    template <class BaseEnv>
      using make_stream_env_t = 
        stdexec::__if_c<
          std::is_base_of_v<stream_env_base, BaseEnv>,
          BaseEnv,
          stream_env<stdexec::__x<BaseEnv>>
        >;

    template <class BaseEnv>
      make_stream_env_t<BaseEnv> make_stream_env(BaseEnv&& base, std::optional<cudaStream_t> stream) noexcept {
        return make_stream_env_t<BaseEnv>{{stream}, (BaseEnv&&)base};
      }

    template <class S>
      concept stream_sender =
        std::execution::sender<S> &&
        std::is_base_of_v<stream_sender_base, std::decay_t<S>>;

    template <class R>
      concept stream_receiver =
        std::execution::receiver<R> &&
        std::is_base_of_v<stream_receiver_base, std::decay_t<R>>;

    struct stream_op_state_base{};

    template <class EnvId, class VariantId>
      class stream_enqueue_receiver {
        using Env = stdexec::__t<EnvId>;
        using Variant = stdexec::__t<VariantId>;

        Env env_;
        Variant* variant_;
        queue::task_base_t* task_;
        queue::producer_t producer_;

      public:
        template <stdexec::__one_of<std::execution::set_value_t,
                                    std::execution::set_error_t,
                                    std::execution::set_stopped_t> Tag,
                  class... As>
          friend void tag_invoke(Tag tag, stream_enqueue_receiver&& self, As&&... as) noexcept {
            self.variant_->template emplace<decayed_tuple<Tag, As...>>(Tag{}, (As&&)as...);
            self.producer_(self.task_);
          }

        template <stdexec::__decays_to<std::exception_ptr> E>
          friend void tag_invoke(std::execution::set_error_t, stream_enqueue_receiver&& self, E&& e) noexcept {
            // What is `exception_ptr` but death pending
            self.variant_->template emplace<decayed_tuple<std::execution::set_error_t, cudaError_t>>(std::execution::set_error, cudaErrorUnknown);
            self.producer_(self.task_);
          }

        friend Env tag_invoke(std::execution::get_env_t, const stream_enqueue_receiver& self) {
          return self.env_;
        }

        stream_enqueue_receiver(
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
        tag(std::move(receiver), std::move(as)...);
      }

    template <class Receiver, class Variant>
      struct continuation_task_t : queue::task_base_t {
        Receiver receiver_;
        Variant* variant_;
        cudaStream_t stream_;
        cudaError_t status_{cudaSuccess};

        continuation_task_t(Receiver receiver, Variant* variant, cudaStream_t stream) noexcept 
          : receiver_{receiver}
          , variant_{variant}
          , stream_{stream} {
          this->execute_ = [](task_base_t* t) noexcept {
            continuation_task_t &self = *reinterpret_cast<continuation_task_t*>(t);

            visit([&self](auto&& tpl) noexcept {
                ::cuda::std::apply([&self](auto tag, auto&&... as) noexcept {
                  tag(std::move(self.receiver_), std::move(as)...);
                }, std::move(tpl));
            }, std::move(*self.variant_));
          };

          this->free_ = [](task_base_t* t) noexcept {
            continuation_task_t &self = *reinterpret_cast<continuation_task_t*>(t);
            STDEXEC_DBG_ERR(cudaFreeAsync(self.atom_next_, self.stream_));
            STDEXEC_DBG_ERR(cudaFreeHost(t));
          };

          this->next_ = nullptr;

          constexpr std::size_t ptr_size = sizeof(this->atom_next_);
          status_ = STDEXEC_DBG_ERR(cudaMallocAsync(&this->atom_next_, ptr_size, stream_));

          if (status_ == cudaSuccess) {
            status_ = STDEXEC_DBG_ERR(cudaMemsetAsync(this->atom_next_, 0, ptr_size, stream_));
          }
        }
      };

    template <class OuterReceiverId>
      struct operation_state_base_t : stream_op_state_base {
        using outer_receiver_t = stdexec::__t<OuterReceiverId>;

        bool owner_{false};
        cudaStream_t stream_{0};
        void *temp_storage_{nullptr};
        outer_receiver_t receiver_;
        stream_priority priority_;
        cudaError_t status_{cudaSuccess};

        operation_state_base_t(outer_receiver_t receiver, stream_priority priority = stream_priority::normal)
          : receiver_(receiver) 
          , priority_(priority) {
        }

        template <class Tag, class... As>
        void propagate_completion_signal(Tag tag, As&&... as) noexcept {
          if constexpr (stream_receiver<outer_receiver_t>) {
            tag((outer_receiver_t&&)receiver_, (As&&)as...);
          } else {
            continuation_kernel
              <std::decay_t<outer_receiver_t>, Tag, As...>
              <<<1, 1, 0, stream_>>>(
                receiver_, tag, (As&&)as...);
          }
        }

        cudaStream_t allocate() {
          if (stream_ == 0) {
            owner_ = true;
            std::tie(stream_, status_) = create_stream_with_priority(priority_);
          }

          return stream_;
        }

        ~operation_state_base_t() {
          if (owner_) {
            STDEXEC_DBG_ERR(cudaStreamDestroy(stream_));
            stream_ = 0;
            owner_ = false;
          }

          if (temp_storage_) {
            STDEXEC_DBG_ERR(cudaFree(temp_storage_));
          }
        }
      };

    template <class ReceiverId>
      struct propagate_receiver_t : stream_receiver_base {
        operation_state_base_t<ReceiverId>& operation_state_;

        template <stdexec::__one_of<std::execution::set_value_t,
                                    std::execution::set_error_t,
                                    std::execution::set_stopped_t> Tag,
                  class... As >
        friend void tag_invoke(Tag tag, propagate_receiver_t&& self, As&&... as) noexcept {
          self.operation_state_.template propagate_completion_signal<Tag, As...>(tag, (As&&)as...);
        }

        friend std::execution::env_of_t<stdexec::__t<ReceiverId>>
        tag_invoke(std::execution::get_env_t, const propagate_receiver_t& self) {
          return std::execution::get_env(self.operation_state_.receiver_);
        }
      };

    template <class SenderId, class InnerReceiverId, class OuterReceiverId>
      struct operation_state_t : operation_state_base_t<OuterReceiverId> {
        using sender_t = stdexec::__t<SenderId>;
        using inner_receiver_t = stdexec::__t<InnerReceiverId>;
        using outer_receiver_t = stdexec::__t<OuterReceiverId>;
        using env_t = std::execution::env_of_t<outer_receiver_t>;
        using variant_t = variant_storage_t<sender_t, env_t>;

        using task_t = continuation_task_t<inner_receiver_t, variant_t>;
        using intermediate_receiver = stdexec::__t<std::conditional_t<
          stream_sender<sender_t>,
          stdexec::__x<inner_receiver_t>,
          stdexec::__x<stream_enqueue_receiver<stdexec::__x<env_t>, stdexec::__x<variant_t>>>>>;
        using inner_op_state_t = std::execution::connect_result_t<sender_t, intermediate_receiver>;

        friend void tag_invoke(std::execution::start_t, operation_state_t& op) noexcept {
          op.started_.test_and_set(::cuda::std::memory_order::relaxed);
          op.stream_ = op.get_stream();

          if (op.status_ != cudaSuccess) {
            // Couldn't allocate memory for operation state, complete with error
            op.propagate_completion_signal(std::execution::set_error, std::move(op.status_));
            return;
          }

          if constexpr (stream_receiver<inner_receiver_t>) {
            if (inner_receiver_t::memory_allocation_size) {
              if (cudaError_t status = 
                    STDEXEC_DBG_ERR(cudaMallocManaged(&op.temp_storage_, inner_receiver_t::memory_allocation_size)); 
                    status != cudaSuccess) {
                // Couldn't allocate memory for intermediate receiver, complete with error
                op.propagate_completion_signal(std::execution::set_error, std::move(status));
                return;
              }
            }
          }

          std::execution::start(op.inner_op_);
        }

        cudaStream_t get_stream() {
          cudaStream_t stream{};
          std::optional<cudaStream_t> env_stream = 
            ::nvexec::STDEXEC_STREAM_DETAIL_NS::get_stream(stdexec::get_env(this->receiver_));

          if (env_stream) {
            stream = *env_stream;
          } else {
            if constexpr (std::is_base_of_v<stream_op_state_base, inner_op_state_t>) {
              stream = inner_op_.get_stream();
            } else {
              stream = this->allocate();
            }
          }

          return stream;
        }

        template <stdexec::__decays_to<outer_receiver_t> OutR, class ReceiverProvider>
          requires stream_sender<sender_t>
        operation_state_t(sender_t&& sender, OutR&& out_receiver, ReceiverProvider receiver_provider, context_state_t context_state)
          : operation_state_base_t<OuterReceiverId>((outer_receiver_t&&)out_receiver, context_state.priority_)
          , inner_op_{std::execution::connect((sender_t&&)sender, receiver_provider(*this))}
        {}

        template <stdexec::__decays_to<outer_receiver_t> OutR, class ReceiverProvider>
          requires (!stream_sender<sender_t>)
        operation_state_t(sender_t&& sender, OutR&& out_receiver, ReceiverProvider receiver_provider, context_state_t context_state)
          : operation_state_base_t<OuterReceiverId>((outer_receiver_t&&)out_receiver, context_state.priority_)
          , storage_(queue::make_host<variant_t>(this->status_))
          , task_(queue::make_host<task_t>(this->status_, receiver_provider(*this), storage_.get(), get_stream()).release())
          , started_(ATOMIC_FLAG_INIT)
          , inner_op_{
              std::execution::connect((sender_t&&)sender,
              stream_enqueue_receiver<stdexec::__x<env_t>, stdexec::__x<variant_t>>{
                std::execution::get_env(out_receiver), storage_.get(), task_, context_state.hub_->producer()})} {
          if (this->status_ == cudaSuccess) {
            this->status_ = task_->status_;
          }
        }

        ~operation_state_t() {
          if (!started_.test(::cuda::memory_order_relaxed)) {
            if (task_) {
              task_->free_(task_);
            }
          }
        }

        STDEXEC_IMMOVABLE(operation_state_t);

        queue::host_ptr<variant_t> storage_;
        task_t *task_{};

        ::cuda::std::atomic_flag started_;
        inner_op_state_t inner_op_;
      };

    template <class Sender, class OuterReceiver>
        requires stream_receiver<OuterReceiver>
      using exit_operation_state_t 
        = operation_state_t<
            stdexec::__x<Sender>, 
            stdexec::__x<propagate_receiver_t<stdexec::__x<OuterReceiver>>>, 
            stdexec::__x<OuterReceiver>>;

    template <class Sender, class OuterReceiver>
      exit_operation_state_t<Sender, OuterReceiver>
      exit_op_state(Sender&& sndr, OuterReceiver&& rcvr, context_state_t context_state) noexcept {
        using ReceiverId = stdexec::__x<OuterReceiver>;
        return exit_operation_state_t<Sender, OuterReceiver>(
          (Sender&&)sndr, 
          (OuterReceiver&&)rcvr, 
          [](operation_state_base_t<ReceiverId>& op) -> propagate_receiver_t<ReceiverId> {
            return propagate_receiver_t<ReceiverId>{{}, op};
          },
          context_state);
      }

    template <class S>
      concept stream_completing_sender =
        std::execution::sender<S> &&
        requires (const S& sndr) {
          { std::execution::get_completion_scheduler<std::execution::set_value_t>(sndr).context_state_ } -> 
              std::same_as<context_state_t>;
        };

    template <class R>
      concept receiver_with_stream_env =
        std::execution::receiver<R> &&
        requires (const R& rcvr) {
          { std::execution::get_scheduler(std::execution::get_env(rcvr)).context_state_ } -> 
              std::same_as<context_state_t>;
        };

    template <class Sender, class InnerReceiver, class OuterReceiver>
      using stream_op_state_t = operation_state_t<stdexec::__x<Sender>,
                                                  stdexec::__x<InnerReceiver>,
                                                  stdexec::__x<OuterReceiver>>;

    template <stream_completing_sender Sender, class OuterReceiver, class ReceiverProvider>
      stream_op_state_t<Sender, std::invoke_result_t<ReceiverProvider, operation_state_base_t<stdexec::__x<OuterReceiver>>&>, OuterReceiver>
      stream_op_state(Sender&& sndr, OuterReceiver&& out_receiver, ReceiverProvider receiver_provider) {
        auto sch = std::execution::get_completion_scheduler<std::execution::set_value_t>(sndr);
        context_state_t context_state = sch.context_state_;

        return stream_op_state_t<
          Sender,
          std::invoke_result_t<ReceiverProvider, operation_state_base_t<stdexec::__x<OuterReceiver>>&>,
          OuterReceiver>(
            (Sender&&)sndr,
            (OuterReceiver&&)out_receiver, 
            receiver_provider,
            context_state);
      }

    template <class Sender, class OuterReceiver, class ReceiverProvider>
      stream_op_state_t<Sender, std::invoke_result_t<ReceiverProvider, operation_state_base_t<stdexec::__x<OuterReceiver>>&>, OuterReceiver>
      stream_op_state(Sender&& sndr, OuterReceiver&& out_receiver, ReceiverProvider receiver_provider, context_state_t context_state) {
        return stream_op_state_t<
          Sender,
          std::invoke_result_t<ReceiverProvider, operation_state_base_t<stdexec::__x<OuterReceiver>>&>, OuterReceiver>(
            (Sender&&)sndr,
            (OuterReceiver&&)out_receiver, 
            receiver_provider, 
            context_state);
      }
  }
}

