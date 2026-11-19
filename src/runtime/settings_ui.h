#pragma once
// Standalone dark-mode Graphics, Display & Audio Settings window.
// Accessible via F1 in-game, --settings CLI flag, or the Settings button in CellForge-Setup.

#include <filesystem>

namespace cfg {

// Opens the settings dialog (blocking until closed). Reads & writes `options_ini`,
// updates cfg::get() and sync_atomics() immediately on Save & Apply.
// Returns true if the user saved changes.
bool show_settings_dialog(const std::filesystem::path& options_ini, void* owner_hwnd = nullptr);

// Non-blocking variant for in-game F1 hotkey (prevents opening multiple instances).
void toggle_settings_dialog_async(const std::filesystem::path& options_ini, void* owner_hwnd = nullptr);

}  // namespace cfg
