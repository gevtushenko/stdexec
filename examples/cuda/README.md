## Stream

The behavior or stream context depending on the previous sender and next receiver:

| sender\receiver | stream                                                       | unknown                                                      |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| stream          | launch a kernel asynchronously                               | pass out receiver into a separate kernel that's asynchronously launched after the sender's one |
| unknown         | connect unknown sender with a receiver that puts data into a device memory stored in the operation state. Launch kernel reading from that memory in stream order. |                                                              |



### Supported senders

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
- [ ] `ensure_started`
- [x] `start_detached`

### TO DO

- [ ] Tests
- [x] NVHPC
- [ ] Executing unknown senders on GPU

