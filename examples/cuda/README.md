### Idea

When `a_sender` precedes `stream_sender` in the pipeline `a_sender | stream_sender`, operation state produced by connecting `stream_sender` with `out_reciever` represents a task in the task queue. The queue is managed by a dedicated thread that polls on the queue using something similar to the run loop. The `a_sender` is connected with a `sink_receiver` that has a pointer to the mentioned operation state and the queue. When `a_receiver` completes on `sink_receiver`, it enqueues operation state in the queue.  When task is executed, it completes on the `stream_receiver` which enqueues more work and completes on `out_receiver`.

When `stream_sender` precedes another `stream_sender` in the pipeline `stream_sender | stream_sender`, first `stream_receiver` executes work on GPU and then completes on the successor `stream_receiver` on CPU. 

### Building blocks

- [ ] GPU/CPU task queue
- [ ] Emplacement into `variant` from GPU
- [ ] Emplacement into `tuple` from GPU

### Supported

- [x] `then`
- [x] `bulk`
- [x] `transfer_when_all_with_variant`
- [x] `transfer_when_all`
- [x] `when_all_with_variant`
- [x] `when_all`
- [x] `split`
- [x] `transfer_just`
- [x] `upon_stopped`
- [x] `upon_error`
- [x] `let_value`
- [x] `let_error`
- [x] `let_stopped`
- [x] `schedule_from`
- [x] `transfer`
- [x] `sync_wait`
- [ ] `ensure_started`
- [x] `start_detached`

### TO DO

- [ ] Tests
- [x] NVHPC
- [ ] Executing unknown senders on GPU

