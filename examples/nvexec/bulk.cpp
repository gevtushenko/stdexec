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

#include <nvexec/stream_context.cuh>
#include <stdexec/execution.hpp>

#include <cstdio>

namespace ex = stdexec;

int main() {
  using nvexec::is_on_gpu;

  nvexec::stream_context stream_ctx{};
  ex::scheduler auto sch = stream_ctx.get_scheduler();

  auto task = stdexec::on(sch, stdexec::just());
  stdexec::sync_wait(std::move(task));
}
