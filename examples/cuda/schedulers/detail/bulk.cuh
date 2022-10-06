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

#include <execution.hpp>
#include <type_traits>

#include "common.cuh"

namespace example::cuda::stream {

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
    class receiver_t : public receiver_base_t {
      using Receiver = _P2300::__t<ReceiverId>;

      Shape shape_;
      Fun f_;

      operation_state_base_t<ReceiverId>& op_state_;

    public:
      template <class... As _NVCXX_CAPTURE_PACK(As)>
        friend void tag_invoke(std::execution::set_value_t, receiver_t&& self, As&&... as)
          noexcept requires _P2300::__callable<Fun, Shape, As...> {
          operation_state_base_t<ReceiverId> &op_state = self.op_state_;

          _NVCXX_EXPAND_PACK(As, as,
            if (self.shape_) {
              cudaStream_t stream = op_state.stream_;
              constexpr int block_threads = 256;
              const int grid_blocks = (static_cast<int>(self.shape_) + block_threads - 1) / block_threads;
              kernel
                <block_threads, Shape, Fun, As...>
                  <<<grid_blocks, block_threads, 0, stream>>>(
                    self.shape_, self.f_, (As&&)as...);
            }

            if (cudaError_t status = STDEXEC_DBG_ERR(cudaPeekAtLastError()); status == cudaSuccess) {
              op_state.propagate_completion_signal(std::execution::set_value, (As&&)as...);
            } else {
              op_state.propagate_completion_signal(std::execution::set_error, std::move(status));
            }
          );
        }

      template <_P2300::__one_of<std::execution::set_error_t,
                              std::execution::set_stopped_t> Tag, class... As>
        friend void tag_invoke(Tag tag, receiver_t&& self, As&&... as) noexcept {
          self.op_state_.propagate_completion_signal(tag, (As&&)as...);
        }

      friend std::execution::env_of_t<Receiver> tag_invoke(std::execution::get_env_t, const receiver_t& self) {
        return std::execution::get_env(self.op_state_.receiver_);
      }

      explicit receiver_t(Shape shape, Fun fun, operation_state_base_t<ReceiverId>& op_state)
        : shape_(shape)
        , f_((Fun&&) fun)
        , op_state_(op_state)
      {}
    };
}

template <class SenderId, std::integral Shape, class FunId>
  struct bulk_sender_t : sender_base_t {
    using Sender = _P2300::__t<SenderId>;
    using Fun = _P2300::__t<FunId>;

    Sender sndr_;
    Shape shape_;
    Fun fun_;

    using set_error_t =
      std::execution::completion_signatures<
        std::execution::set_error_t(cudaError_t)>;

    template <class Receiver>
      using receiver_t = bulk::receiver_t<_P2300::__x<Receiver>, Shape, Fun>;

    template <class... Tys>
    using set_value_t =
      std::execution::completion_signatures<
        std::execution::set_value_t(Tys...)>;

    template <class Self, class Env>
      using completion_signatures =
        std::execution::__make_completion_signatures<
          _P2300::__member_t<Self, Sender>,
          Env,
          set_error_t,
          _P2300::__q<set_value_t>>;

    template <_P2300::__decays_to<bulk_sender_t> Self, std::execution::receiver Receiver>
      requires std::execution::receiver_of<Receiver, completion_signatures<Self, std::execution::env_of_t<Receiver>>>
    friend auto tag_invoke(std::execution::connect_t, Self&& self, Receiver&& rcvr)
      -> stream_op_state_t<_P2300::__member_t<Self, Sender>, receiver_t<Receiver>, Receiver> {
        return stream_op_state<_P2300::__member_t<Self, Sender>>(
            ((Self&&)self).sndr_,
            (Receiver&&)rcvr,
            [&](operation_state_base_t<_P2300::__x<Receiver>>& stream_provider) -> receiver_t<Receiver> {
              return receiver_t<Receiver>(self.shape_, self.fun_, stream_provider);
            });
      }

    template <_P2300::__decays_to<bulk_sender_t> Self, class Env>
    friend auto tag_invoke(std::execution::get_completion_signatures_t, Self&&, Env)
      -> std::execution::dependent_completion_signatures<Env>;

    template <_P2300::__decays_to<bulk_sender_t> Self, class Env>
    friend auto tag_invoke(std::execution::get_completion_signatures_t, Self&&, Env)
      -> completion_signatures<Self, Env> requires true;

    template <std::execution::tag_category<std::execution::forwarding_sender_query> Tag, class... As>
      requires _P2300::__callable<Tag, const Sender&, As...>
    friend auto tag_invoke(Tag tag, const bulk_sender_t& self, As&&... as)
      noexcept(_P2300::__nothrow_callable<Tag, const Sender&, As...>)
      -> _P2300::__call_result_if_t<std::execution::tag_category<Tag, std::execution::forwarding_sender_query>, Tag, const Sender&, As...> {
      return ((Tag&&) tag)(self.sndr_, (As&&) as...);
    }
  };

}

namespace example::cuda::multi_gpu {

namespace bulk {
  template <int BlockThreads, std::integral Shape, class Fun, class... As>
    __launch_bounds__(BlockThreads)
    __global__ void kernel(Shape begin, Shape end, Fun fn, As... as) {
      const Shape i = begin 
                    + static_cast<Shape>(threadIdx.x + blockIdx.x * blockDim.x);

      if (i < end) {
        fn(i, as...);
      }
    }

  template <class SenderId, class ReceiverId, class Shape, class Fun>
    struct operation_t;

  template <class SenderId, class ReceiverId, std::integral Shape, class Fun>
    class receiver_t : public example::cuda::stream::receiver_base_t {
      using Receiver = _P2300::__t<ReceiverId>;

      Shape shape_;
      Fun f_;

      operation_t<SenderId, ReceiverId, Shape, Fun>& op_state_;

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

    public:
      template <class... As _NVCXX_CAPTURE_PACK(As)>
        friend void tag_invoke(std::execution::set_value_t, receiver_t&& self, As&&... as)
          noexcept requires _P2300::__callable<Fun, Shape, As...> {
          operation_t<SenderId, ReceiverId, Shape, Fun> &op_state = self.op_state_;

          // TODO Usual logic when there's only a single GPU
          cudaStream_t baseline_stream = op_state.stream_;
          cudaEventRecord(op_state.ready_to_launch_, baseline_stream);

          for (int dev = 0; dev < op_state.num_devices_; dev++) {
            cudaStreamWaitEvent(op_state.streams_[dev], op_state.ready_to_launch_);
          }

          _NVCXX_EXPAND_PACK(As, as,
            if (self.shape_) {
              constexpr int block_threads = 256;
              for (int dev = 0; dev < op_state.num_devices_; dev++) {
                auto [begin, end] = even_share(self.shape_, dev, op_state.num_devices_);
                auto shape = static_cast<int>(end - begin);
                const int grid_blocks = (shape + block_threads - 1) / block_threads;

                if (begin < end) {
                  cudaSetDevice(dev);
                  kernel
                    <block_threads, Shape, Fun, As...>
                      <<<grid_blocks, block_threads, 0, op_state.streams_[dev]>>>(
                        begin, end, self.f_, (As&&)as...);
                  cudaEventRecord(op_state.ready_to_complete_[dev], op_state.streams_[dev]);
                }
              }
            }

            cudaSetDevice(op_state.current_device_);
            for (int dev = 0; dev < op_state.num_devices_; dev++) {
              cudaStreamWaitEvent(baseline_stream, op_state.ready_to_complete_[dev]);
            }

            if (cudaError_t status = STDEXEC_DBG_ERR(cudaPeekAtLastError()); status == cudaSuccess) {
              op_state.propagate_completion_signal(std::execution::set_value, (As&&)as...);
            } else {
              op_state.propagate_completion_signal(std::execution::set_error, std::move(status));
            }
          );
        }

      template <_P2300::__one_of<std::execution::set_error_t,
                              std::execution::set_stopped_t> Tag, class... As>
        friend void tag_invoke(Tag tag, receiver_t&& self, As&&... as) noexcept {
          self.op_state_.propagate_completion_signal(tag, (As&&)as...);
        }

      friend std::execution::env_of_t<Receiver> tag_invoke(std::execution::get_env_t, const receiver_t& self) {
        return std::execution::get_env(self.op_state_.receiver_);
      }

      explicit receiver_t(Shape shape, Fun fun, operation_t<SenderId, ReceiverId, Shape, Fun>& op_state)
        : shape_(shape)
        , f_((Fun&&) fun)
        , op_state_(op_state)
      {}
    };

  template <class SenderId, class ReceiverId, class Shape, class Fun>
    using operation_base_t =
      example::cuda::stream::detail::operation_state_t<
        SenderId,
        _P2300::__x<receiver_t<SenderId, ReceiverId, Shape, Fun>>,
        ReceiverId>;

  template <class SenderId, class ReceiverId, class Shape, class Fun>
    struct operation_t : operation_base_t<SenderId, ReceiverId, Shape, Fun> {
      using Sender = _P2300::__t<SenderId>;
      using Receiver = _P2300::__t<ReceiverId>;

      template <class _Receiver2>
        operation_t(
            int num_devices,
            Sender&& __sndr, 
            _Receiver2&& __rcvr, 
            Shape shape, 
            Fun fun)
          : operation_base_t<SenderId, ReceiverId, Shape, Fun>(
              (Sender&&) __sndr,
              std::execution::get_completion_scheduler<std::execution::set_value_t>(__sndr).hub_,
              (_Receiver2&&)__rcvr,
              [&] (example::cuda::stream::operation_state_base_t<_P2300::__x<_Receiver2>> &) -> receiver_t<SenderId, ReceiverId, Shape, Fun> {
                return receiver_t<SenderId, ReceiverId, Shape, Fun>(shape, fun, *this);
              })
          , num_devices_(num_devices)
          , streams_(new cudaStream_t[num_devices_]) 
          , ready_to_complete_(new cudaEvent_t[num_devices_])
        {
          cudaGetDevice(&current_device_);
          cudaEventCreate(&ready_to_launch_);
          for (int dev = 0; dev < num_devices_; dev++) {
            cudaSetDevice(dev);
            cudaStreamCreate(streams_.get() + dev);
            cudaEventCreate(ready_to_complete_.get() + dev);
          }
          cudaSetDevice(current_device_);
        }

      ~operation_t() {
        for (int dev = 0; dev < num_devices_; dev++) {
          cudaSetDevice(dev);
          cudaStreamDestroy(streams_[dev]);
          cudaEventDestroy(ready_to_complete_[dev]);
        }
        cudaSetDevice(current_device_);
        cudaEventDestroy(ready_to_launch_);
      }

      _P2300_IMMOVABLE(operation_t);

      int num_devices_{};
      int current_device_{};
      std::unique_ptr<cudaStream_t[]> streams_;
      std::unique_ptr<cudaEvent_t[]> ready_to_complete_;
      cudaEvent_t ready_to_launch_;
    };
}

template <class SenderId, std::integral Shape, class FunId>
  struct bulk_sender_t : example::cuda::stream::sender_base_t {
    using Sender = _P2300::__t<SenderId>;
    using Fun = _P2300::__t<FunId>;

    int num_devices_;
    Sender sndr_;
    Shape shape_;
    Fun fun_;

    using set_error_t =
      std::execution::completion_signatures<
        std::execution::set_error_t(cudaError_t)>;

    template <class Receiver>
      using receiver_t = bulk::receiver_t<SenderId, _P2300::__x<Receiver>, Shape, Fun>;

    template <class... Tys>
    using set_value_t =
      std::execution::completion_signatures<
        std::execution::set_value_t(Tys...)>;

    template <class Self, class Env>
      using completion_signatures =
        std::execution::__make_completion_signatures<
          _P2300::__member_t<Self, Sender>,
          Env,
          set_error_t,
          _P2300::__q<set_value_t>>;

    template <_P2300::__decays_to<bulk_sender_t> Self, std::execution::receiver Receiver>
      requires std::execution::receiver_of<Receiver, completion_signatures<Self, std::execution::env_of_t<Receiver>>>
    friend auto tag_invoke(std::execution::connect_t, Self&& self, Receiver&& rcvr)
      -> bulk::operation_t<_P2300::__x<std::__member_t<Self, Sender>>, _P2300::__x<Receiver>, Shape, Fun> {
      return bulk::operation_t<_P2300::__x<std::__member_t<Self, Sender>>, _P2300::__x<Receiver>, Shape, Fun>(
          self.num_devices_,
          ((Self&&)self).sndr_,
          (Receiver&&)rcvr,
          self.shape_,
          self.fun_);
      }

    template <_P2300::__decays_to<bulk_sender_t> Self, class Env>
    friend auto tag_invoke(std::execution::get_completion_signatures_t, Self&&, Env)
      -> std::execution::dependent_completion_signatures<Env>;

    template <_P2300::__decays_to<bulk_sender_t> Self, class Env>
    friend auto tag_invoke(std::execution::get_completion_signatures_t, Self&&, Env)
      -> completion_signatures<Self, Env> requires true;

    template <std::execution::tag_category<std::execution::forwarding_sender_query> Tag, class... As>
      requires _P2300::__callable<Tag, const Sender&, As...>
    friend auto tag_invoke(Tag tag, const bulk_sender_t& self, As&&... as)
      noexcept(_P2300::__nothrow_callable<Tag, const Sender&, As...>)
      -> _P2300::__call_result_if_t<std::execution::tag_category<Tag, std::execution::forwarding_sender_query>, Tag, const Sender&, As...> {
      return ((Tag&&) tag)(self.sndr_, (As&&) as...);
    }
  };

}

