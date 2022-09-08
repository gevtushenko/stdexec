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

#include <execution.hpp>
#include "schedulers/stream.cuh"
#include <thrust/device_vector.h>

namespace ex = std::execution;

enum class device_type {
  host,
  device
};

#ifdef _NVHPC_CUDA
#include <nv/target>

__host__ __device__ inline device_type get_device_type() {
  if target (nv::target::is_host) {
    return device_type::host;
  }
  else {
    return device_type::device;
  }
}
#elif defined(__clang__) && defined(__CUDA__)
__host__ inline device_type get_device_type() { return device_type::host; }
__device__ inline device_type get_device_type() { return device_type::device; }
#endif

inline __host__ __device__ bool is_on_gpu() {
  return get_device_type() == device_type::device;
}

struct printer_t {
  int val_;

  __host__ __device__ void operator()(int idx, int val) {
    printf("%d: %d (on %s) val = %d\n", val_, idx, (is_on_gpu() ? "gpu" : "cpu"), val);
  }
};

__host__ __device__ 
std::uint64_t fib(std::uint64_t n) {
   if (n <= 1) {
     return n;
   }

   return fib(n - 1) + fib(n - 2);
}

int main() {
  example::cuda::stream::scheduler_t scheduler{};

  if (0)
  {
    auto snd = ex::schedule(scheduler) 
             | ex::then([]() -> int { return 42; })
             | ex::bulk(4, printer_t{1})
             | ex::bulk(4, printer_t{2});
    std::this_thread::sync_wait(std::move(snd));
  }

  {
    auto snd = ex::just_stopped() 
             | ex::transfer(scheduler)
             | ex::upon_stopped([] () { 
                 if (is_on_gpu()) {
                   std::printf("+\n");
                 }
               });
    std::this_thread::sync_wait(std::move(snd));
  }

  if (0)
  {
    const int n = 1024;
    thrust::device_vector<std::uint64_t> a(n, 10);
    thrust::device_vector<std::uint64_t> b(n);
    thrust::device_vector<std::uint64_t> c(n);
    thrust::device_vector<std::uint64_t> d(n);

    std::uint64_t *d_a = thrust::raw_pointer_cast(a.data());
    std::uint64_t *d_b = thrust::raw_pointer_cast(b.data());
    std::uint64_t *d_c = thrust::raw_pointer_cast(c.data());
    std::uint64_t *d_d = thrust::raw_pointer_cast(d.data());

    auto snd = 
      ex::when_all(
        ex::schedule(scheduler) | ex::bulk(n, [d_a, d_b](int idx){ d_b[idx] += fib(d_a[idx]); }),
        ex::schedule(scheduler) | ex::bulk(n, [d_a, d_c](int idx){ d_c[idx] -= fib(d_a[idx]); }),
        ex::schedule(scheduler) | ex::bulk(n, [d_a, d_d](int idx){ d_d[idx] *= fib(d_a[idx]); }))
      | ex::transfer(scheduler) | ex::then([] { std::printf("done\n"); });

    std::this_thread::sync_wait(std::move(snd));
  }
}

