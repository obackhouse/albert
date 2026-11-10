#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace albert {

struct TensorKey {
    std::string symbol;
    std::vector<std::string> spaces;

    bool operator==(const TensorKey& other) const {
        return symbol == other.symbol && spaces == other.spaces;
    }
};

struct TensorKeyHash {
    std::size_t operator()(const TensorKey& key) const noexcept;
};

}  // namespace albert
