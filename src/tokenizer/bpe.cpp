#include "tokenizer/bpe.h"

#include <algorithm>
#include <queue>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "tokenizer/unicode.h"

namespace ography {

namespace {

// GPT-2's bytes_to_unicode: printable bytes map to themselves, the rest to U+0100 onwards,
// so every token is a printable string.
void build_byte_map(std::string (&byte_to_unicode)[256]) {
  int next = 256;
  for (int b = 0; b < 256; ++b) {
    const bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE);
    std::string s;
    unicode::append_utf8(printable ? char32_t(b) : char32_t(next++), s);
    byte_to_unicode[b] = std::move(s);
  }
}

}  // namespace

BpeModel::BpeModel(const nlohmann::json& tj) {
  if (!tj.at("normalizer").is_null() && tj["normalizer"].value("type", "") != "NFC") {
    throw std::runtime_error("tokenizer: unsupported normalizer: " + tj["normalizer"].dump());
  }
  // NFC (Qwen2): input is assumed to be NFC already, which holds for nearly all real text.
  // TODO: implement NFC normalization if non-NFC input turns up.

  pretokenizer_ = detect_pretokenizer(tj.at("pre_tokenizer"));
  const auto& model = tj.at("model");
  if (model.value("type", "BPE") != "BPE") {
    throw std::runtime_error("tokenizer: model type must be BPE");
  }
  ignore_merges_ = model.value("ignore_merges", false);

  build_byte_map(byte_to_unicode_);
  // Reverse map: the stand-in code point -> original byte.
  std::unordered_map<char32_t, unsigned char> unicode_to_byte;
  for (int b = 0; b < 256; ++b) {
    char32_t cp;
    unicode::decode_utf8(byte_to_unicode_[b], 0, cp);
    unicode_to_byte[cp] = static_cast<unsigned char>(b);
  }

  int32_t max_id = -1;
  for (const auto& [token, id] : model.at("vocab").items()) {
    vocab_[token] = id.get<int32_t>();
    max_id = std::max(max_id, id.get<int32_t>());
  }
  for (const auto& t : tj.at("added_tokens")) {
    added_[t.at("content").get<std::string>()] = t.at("id").get<int32_t>();
    max_id = std::max(max_id, t.at("id").get<int32_t>());
  }

  id_to_bytes_.resize(max_id + 1);
  for (const auto& [token, id] : vocab_) {
    std::string bytes;
    for (size_t i = 0; i < token.size();) {
      char32_t cp;
      i += unicode::decode_utf8(token, i, cp);
      const auto it = unicode_to_byte.find(cp);
      if (it == unicode_to_byte.end()) {
        throw std::runtime_error("tokenizer: vocab token is not byte-level encoded: " + token);
      }
      bytes += static_cast<char>(it->second);
    }
    id_to_bytes_[id] = std::move(bytes);
  }
  added_by_first_byte_.resize(256);
  for (const auto& [content, id] : added_) {
    id_to_bytes_[id] = content;
    if (!content.empty()) {
      added_by_first_byte_[static_cast<unsigned char>(content[0])].emplace_back(content, id);
    }
  }
  for (auto& bucket : added_by_first_byte_) {
    std::sort(bucket.begin(), bucket.end(),
              [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
  }

  for (int b = 0; b < 256; ++b) {
    const auto it = vocab_.find(byte_to_unicode_[b]);
    if (it == vocab_.end()) throw std::runtime_error("tokenizer: vocab lacks a byte token");
    byte_ids_[b] = it->second;
  }

  // Merges are either "left right" strings or [left, right] pairs; rank = position.
  const auto& merges = model.at("merges");
  merges_.reserve(merges.size());
  for (size_t rank = 0; rank < merges.size(); ++rank) {
    std::string left, right;
    if (merges[rank].is_string()) {
      const std::string m = merges[rank].get<std::string>();
      const size_t sp = m.find(' ');
      if (sp == std::string::npos) throw std::runtime_error("tokenizer: bad merge: " + m);
      left = m.substr(0, sp);
      right = m.substr(sp + 1);
    } else {
      left = merges[rank].at(0).get<std::string>();
      right = merges[rank].at(1).get<std::string>();
    }
    const auto l = vocab_.find(left), r = vocab_.find(right), lr = vocab_.find(left + right);
    if (l == vocab_.end() || r == vocab_.end() || lr == vocab_.end()) {
      throw std::runtime_error("tokenizer: merge refers to unknown tokens: " + left + " " + right);
    }
    merges_.try_emplace(pair_key(l->second, r->second),
                        Merge{static_cast<int32_t>(rank), lr->second});
  }
}

int32_t BpeModel::token_id(std::string_view token) const {
  const std::string key(token);
  if (const auto it = added_.find(key); it != added_.end()) return it->second;
  if (const auto it = vocab_.find(key); it != vocab_.end()) return it->second;
  return -1;
}

void BpeModel::encode(std::string_view text, std::vector<int32_t>& out) const {
  size_t plain_start = 0;
  for (size_t i = 0; i < text.size();) {
    const auto& bucket = added_by_first_byte_[static_cast<unsigned char>(text[i])];
    const auto it = std::find_if(bucket.begin(), bucket.end(), [&](const auto& t) {
      return text.compare(i, t.first.size(), t.first) == 0;
    });
    if (it == bucket.end()) {
      ++i;
      continue;
    }
    encode_ordinary(text.substr(plain_start, i - plain_start), out);
    out.push_back(it->second);
    i += it->first.size();
    plain_start = i;
  }
  encode_ordinary(text.substr(plain_start), out);
}

void BpeModel::encode_ordinary(std::string_view text, std::vector<int32_t>& out) const {
  std::vector<std::string_view> pieces;
  pretokenize(pretokenizer_, text, pieces);
  for (std::string_view p : pieces) bpe(p, out);
}

// Standard BPE on one pretokenized piece: repeatedly merge the adjacent pair with the lowest
// rank (leftmost first on ties), using a heap with lazy invalidation, O(n log n).
void BpeModel::bpe(std::string_view piece, std::vector<int32_t>& out) const {
  if (piece.empty()) return;
  if (piece.size() == 1) {
    out.push_back(byte_ids_[static_cast<unsigned char>(piece[0])]);
    return;
  }
  if (ignore_merges_) {
    std::string mapped;
    for (char c : piece) mapped += byte_to_unicode_[static_cast<unsigned char>(c)];
    if (const auto it = vocab_.find(mapped); it != vocab_.end()) {
      out.push_back(it->second);
      return;
    }
  }

  const int n = static_cast<int>(piece.size());
  std::vector<int32_t> id(n);
  std::vector<int> prev(n), next(n);
  for (int i = 0; i < n; ++i) {
    id[i] = byte_ids_[static_cast<unsigned char>(piece[i])];
    prev[i] = i - 1;
    next[i] = i + 1 < n ? i + 1 : -1;
  }

  struct Candidate {
    int32_t rank;
    int pos;  // index of the left symbol
    int32_t left, right;
    bool operator>(const Candidate& o) const {
      return rank != o.rank ? rank > o.rank : pos > o.pos;
    }
  };
  std::priority_queue<Candidate, std::vector<Candidate>, std::greater<>> heap;
  auto push = [&](int pos) {
    if (pos < 0 || next[pos] < 0) return;
    const auto it = merges_.find(pair_key(id[pos], id[next[pos]]));
    if (it != merges_.end()) heap.push({it->second.rank, pos, id[pos], id[next[pos]]});
  };
  for (int i = 0; i + 1 < n; ++i) push(i);

  while (!heap.empty()) {
    const Candidate c = heap.top();
    heap.pop();
    const int r = next[c.pos];
    // Stale if either symbol was merged away or changed since this entry was pushed.
    if (id[c.pos] != c.left || r < 0 || id[r] != c.right) continue;
    id[c.pos] = merges_.at(pair_key(c.left, c.right)).merged_id;
    id[r] = -1;
    next[c.pos] = next[r];
    if (next[r] >= 0) prev[next[r]] = c.pos;
    push(prev[c.pos]);
    push(c.pos);
  }

  for (int i = 0; i >= 0; i = next[i]) out.push_back(id[i]);
}

}  // namespace ography
