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

#include "common.cuh"

namespace _P2300::this_thread {
  namespace stream_sync_wait {
    template <class _Sender>
      using __into_variant_result_t =
        decltype(execution::into_variant(__declval<_Sender>()));

    struct __env {
      execution::run_loop::__scheduler __sched_;

      friend auto tag_invoke(execution::get_scheduler_t, const __env& __self) noexcept
        -> execution::run_loop::__scheduler {
        return __self.__sched_;
      }

      friend auto tag_invoke(execution::get_delegatee_scheduler_t, const __env& __self) noexcept
        -> execution::run_loop::__scheduler {
        return __self.__sched_;
      }
    };

    // What should sync_wait(just_stopped()) return?
    template <class _Sender>
        requires execution::sender<_Sender, __env>
      using __sync_wait_result_t =
        execution::value_types_of_t<
          _Sender,
          __env,
          execution::__decayed_tuple,
          __single_t>;

    template <class _Sender>
      using __sync_wait_with_variant_result_t =
        __sync_wait_result_t<__into_variant_result_t<_Sender>>;

    template <class _SenderId>
      struct __state;

    template <class _SenderId>
      struct __receiver : example::cuda::stream::receiver_base_t {
        using _Sender = __t<_SenderId>;
        __state<_SenderId>* __state_;
        execution::run_loop* __loop_;
        template <class _Error>
        void __set_error(_Error __err) noexcept {
          cudaDeviceSynchronize(); // TODO Synchronize stream
          if constexpr (__decays_to<_Error, std::exception_ptr>)
            __state_->__data_.template emplace<2>((_Error&&) __err);
          else if constexpr (__decays_to<_Error, std::error_code>)
            __state_->__data_.template emplace<2>(std::make_exception_ptr(std::system_error(__err)));
          else
            __state_->__data_.template emplace<2>(std::make_exception_ptr((_Error&&) __err));
          __loop_->finish();
        }
        template <class _Sender2 = _Sender, class... _As _NVCXX_CAPTURE_PACK(_As)>
          requires constructible_from<__sync_wait_result_t<_Sender2>, _As...>
        friend void tag_invoke(execution::set_value_t, __receiver&& __rcvr, _As&&... __as) noexcept try {
          _NVCXX_EXPAND_PACK(_As, __as,
            cudaDeviceSynchronize(); // TODO Synchronize stream
            __rcvr.__state_->__data_.template emplace<1>((_As&&) __as...);
          )
          __rcvr.__loop_->finish();
        } catch(...) {
          __rcvr.__set_error(std::current_exception());
        }
        template <class _Error>
        friend void tag_invoke(execution::set_error_t, __receiver&& __rcvr, _Error __err) noexcept {
            cudaDeviceSynchronize(); // TODO Synchronize stream
          __rcvr.__set_error((_Error &&) __err);
        }
        friend void tag_invoke(execution::set_stopped_t __d, __receiver&& __rcvr) noexcept {
            cudaDeviceSynchronize(); // TODO Synchronize stream
          __rcvr.__state_->__data_.template emplace<3>(__d);
          __rcvr.__loop_->finish();
        }
        friend __env
        tag_invoke(execution::get_env_t, const __receiver& __rcvr) noexcept {
          return {__rcvr.__loop_->get_scheduler()};
        }
      };

    template <class _SenderId>
      struct __state {
        using _Tuple = __sync_wait_result_t<__t<_SenderId>>;
        std::variant<std::monostate, _Tuple, std::exception_ptr, execution::set_stopped_t> __data_{};
      };

    template <class _Sender>
      using __into_variant_result_t =
        decltype(execution::into_variant(__declval<_Sender>()));
  }
}

namespace example::cuda::stream {

namespace sync_wait {
template <std::execution::__single_value_variant_sender<_P2300::this_thread::stream_sync_wait::__env> _Sender>
  auto impl(_Sender&& __sndr)
    -> std::optional<_P2300::this_thread::stream_sync_wait::__sync_wait_result_t<_Sender>> {
  using state_t = _P2300::this_thread::stream_sync_wait::__state<std::__x<_Sender>>;
  state_t __state{};
  std::execution::run_loop __loop;

  // Launch the sender with a continuation that will fill in a variant
  // and notify a condition variable.
  auto __op_state = std::execution::connect(
      (_Sender &&) __sndr, _P2300::this_thread::stream_sync_wait::__receiver<std::__x<_Sender>>{{}, &__state, &__loop});
  std::execution::start(__op_state); 

  // Wait for the variant to be filled in.
  __loop.run();

  if (__state.__data_.index() == 2) {
    std::rethrow_exception(std::get<2>(__state.__data_));
  }

  if (__state.__data_.index() == 3) {
    return std::nullopt;
  }

  return std::move(std::get<1>(__state.__data_));
}
}  // namespace sync_wait

}  // namespace example::cuda::stream
