#pragma once

#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>

namespace ography {

class Engine;

// A single worker thread that owns the Engine; HTTP handler threads submit jobs and
// wait. Guarantees one request on the GPU at a time and that CUDA is only ever used
// from one thread.
class RequestQueue {
 public:
  explicit RequestQueue(Engine& engine);
  ~RequestQueue();  // drains and joins

  using Job = std::function<void(Engine&)>;
  void submit(Job job);

 private:
  void worker_loop();

  Engine& engine_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<Job> jobs_;
  bool stopping_ = false;
  std::thread worker_;
};

}  // namespace ography
