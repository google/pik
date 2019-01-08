#include "x11/icc.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>

namespace pik {

namespace {

constexpr char kIccProfileAtomName[] = "_ICC_PROFILE";
constexpr uint32_t kMaxIccProfileSize = 1 << 24;

struct FreeDeleter {
  void operator()(void* const p) const { std::free(p); }
};
template <typename T>
using XcbUniquePtr = std::unique_ptr<T, FreeDeleter>;

}  // namespace

PaddedBytes GetMonitorIccProfile(xcb_connection_t* const connection,
                                 const int screen_number) {
  if (connection == nullptr) {
    return PaddedBytes();
  }

  const xcb_intern_atom_cookie_t atomRequest =
      xcb_intern_atom(connection, /*only_if_exists=*/1,
                      sizeof kIccProfileAtomName - 1, kIccProfileAtomName);
  const XcbUniquePtr<xcb_intern_atom_reply_t> atomReply(
      xcb_intern_atom_reply(connection, atomRequest, nullptr));
  if (atomReply == nullptr) {
    return PaddedBytes();
  }
  const xcb_atom_t iccProfileAtom = atomReply->atom;

  const xcb_screen_t* screen = nullptr;
  int i = 0;
  for (xcb_screen_iterator_t it =
           xcb_setup_roots_iterator(xcb_get_setup(connection));
       it.rem; xcb_screen_next(&it)) {
    if (i == screen_number) {
      screen = it.data;
      break;
    }
    ++i;
  }
  if (screen == nullptr) {
    return PaddedBytes();
  }
  const xcb_get_property_cookie_t profileRequest = xcb_get_property(
      connection, /*_delete=*/0, screen->root, iccProfileAtom,
      XCB_GET_PROPERTY_TYPE_ANY, /*long_offset=*/0, kMaxIccProfileSize);
  const XcbUniquePtr<xcb_get_property_reply_t> profile(
      xcb_get_property_reply(connection, profileRequest, nullptr));
  if (profile == nullptr || profile->bytes_after > 0) {
    return PaddedBytes();
  }

  PaddedBytes result(xcb_get_property_value_length(profile.get()));
  std::copy_n(
      reinterpret_cast<const uint8_t*>(xcb_get_property_value(profile.get())),
      xcb_get_property_value_length(profile.get()), result.begin());
  return result;
}

}  // namespace pik
