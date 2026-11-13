#include "engine/request_queue.h"

#include <exception>

#include "core/logging.h"
#include "engine/engine.h"

namespace ography {

RequestQueue::RequestQueue(Engine& engine) : engine_(engine) {
  worker_ = std::thread([this] { worker_loop(); });
}

RequestQueue::~RequestQueue() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    stopping_ = true;
  }
  cv_.notify_all();
  worker_.join();
}

void RequestQueue::submit(Job job) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    jobs_.push_back(std::move(job));
  }
  cv_.notify_one();
}

void RequestQueue::worker_loop() {
  while (true) {
    Job job;
    {
      std::unique_lock<std::mutex> lock(mu_);
      cv_.wait(lock, [this] { return stopping_ || !jobs_.empty(); });
      if (jobs_.empty()) return;  // stopping, and everything submitted has run
      job = std::move(jobs_.front());
      jobs_.pop_front();
    }
    try {
      job(engine_);
    } catch (const std::exception& e) {
      // Jobs report their own errors; this only guards the worker thread.
      log(LogLevel::Error, std::string("request job threw: ") + e.what());
    }
  }
}

}  // namespace ography
