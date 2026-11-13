#include "engine/request_queue.h"

#include "core/logging.h"
#include "engine/engine.h"

namespace ography {

RequestQueue::RequestQueue(Engine& engine) : engine_(engine) {
  // TODO: worker_ = std::thread([this] { worker_loop(); });
}

RequestQueue::~RequestQueue() {
  // TODO: set stopping_, notify, join worker_.
}

void RequestQueue::submit(Job job) { OGRAPHY_TODO("RequestQueue::submit"); }

void RequestQueue::worker_loop() { OGRAPHY_TODO("RequestQueue::worker_loop"); }

}  // namespace ography
