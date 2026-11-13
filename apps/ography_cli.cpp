// Terminal chat against the engine directly, no HTTP. The quickest way to debug
// tokenizer / chat-template / sampling problems.
//
//   ography_cli --model /path/to/hf-model [--temperature 0.7] [--greedy]
//
// Prints time-to-first-token and tokens/sec after each reply.

#include "core/logging.h"
#include "engine/engine.h"

int main(int argc, char** argv) {
  using namespace ography;
  // TODO: parse args; loop: read line -> append to history -> chat_template().render ->
  //       tokenizer().encode -> engine.generate (print tokens as they stream).
  OGRAPHY_TODO("ography_cli");
}
