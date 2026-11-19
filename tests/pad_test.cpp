// Prints what the runtime sees from the first controller for 20 seconds:
// press buttons and move the sticks to check the mapping.
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>

#include "input.h"

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  using namespace input;
  const struct { int word; uint16_t bit; const char* name; } names[] = {
      {1, SELECT, "Select"}, {1, START, "Start"}, {1, L3, "L3"}, {1, R3, "R3"}, {1, UP, "Up"}, {1, DOWN, "Down"},
      {1, LEFT, "Left"}, {1, RIGHT, "Right"}, {2, CROSS, "Cross"}, {2, CIRCLE, "Circle"}, {2, SQUARE, "Square"},
      {2, TRIANGLE, "Triangle"}, {2, L1, "L1"}, {2, R1, "R1"}, {2, L2, "L2"}, {2, R2, "R2"}};
  std::string last;
  const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(20);
  std::printf("reading controller 0 for 20 s...\n");
  while (std::chrono::steady_clock::now() < end) {
    const PadState s = read(0);
    std::string line;
    if (!s.connected) {
      line = "no controller";
    } else {
      char sticks[96];
      std::snprintf(sticks, sizeof sticks, "L(%3u,%3u) R(%3u,%3u) L2 %3u R2 %3u |", s.lx, s.ly, s.rx, s.ry, s.pressure[10],
                    s.pressure[11]);
      line = sticks;
      for (const auto& n : names)
        if (((n.word == 1 ? s.digital1 : s.digital2) & n.bit) != 0) line += std::string(" ") + n.name;
    }
    if (line != last) {
      std::printf("%s\n", line.c_str());
      last = line;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  return 0;
}
