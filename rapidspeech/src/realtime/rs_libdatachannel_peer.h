#pragma once

#include "rs_realtime_protocol.h"
#include "rs_webrtc_transport.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

#ifdef RS_ENABLE_LIBDATACHANNEL

#include <rtc/rtc.h>

namespace rs::realtime
{

enum class LibdatachannelPeerState : uint8_t
{
    New = 0,
    Connecting = 1,
    Connected = 2,
    Disconnected = 3,
    Failed = 4,
    Closed = 5,
};

struct LibdatachannelConfig
{
    std::vector<std::string> ice_servers;
    std::string reliable_label = "rs-reliable-v1";
    std::string realtime_label = "rs-realtime-v1";
    int max_message_size = 0;
};

struct LibdatachannelCallbacks
{
    std::function<void(const std::string &sdp, const std::string &type)>
        on_local_description;
    std::function<void(const std::string &candidate, const std::string &mid)>
        on_local_candidate;
    std::function<void(LibdatachannelPeerState state)> on_state_change;
    std::function<void(const std::string &message)> on_error;
};

// Owns one libdatachannel PeerConnection and maps two DataChannels into the
// transport-neutral WebrtcTransport. Signaling is intentionally supplied by
// the caller so this class does not impose a WebSocket or HTTP server.
class LibdatachannelPeer final
{
public:
    LibdatachannelPeer(LibdatachannelConfig config,
                       LibdatachannelCallbacks callbacks = {});
    ~LibdatachannelPeer();

    LibdatachannelPeer(const LibdatachannelPeer &) = delete;
    LibdatachannelPeer &operator=(const LibdatachannelPeer &) = delete;

    // When offerer is true, create the two protocol DataChannels and let
    // libdatachannel generate the local offer. An answerer waits for SDP and
    // receives DataChannels through onDataChannel.
    bool start(bool offerer);
    bool set_remote_description(const std::string &sdp, const std::string &type);
    bool add_remote_candidate(const std::string &candidate, const std::string &mid);
    void close();

    bool started() const;
    bool connected() const;
    LibdatachannelPeerState state() const;
    TransportStats stats() const;

    WebrtcTransport &transport();
    const WebrtcTransport &transport() const;

private:
    static void on_local_description(int pc, const char *sdp, const char *type, void *ptr);
    static void on_local_candidate(int pc, const char *candidate, const char *mid, void *ptr);
    static void on_state_change(int pc, rtcState state, void *ptr);
    static void on_data_channel(int pc, int dc, void *ptr);
    static void on_channel_closed(int id, void *ptr);
    static void on_channel_error(int id, const char *error, void *ptr);
    static void on_channel_message(int id, const char *data, int size, void *ptr);

    bool create_protocol_channels();
    bool bind_channel(int id, const std::string &label);
    bool send_channel(int id, const uint8_t *data, size_t size);
    void handle_channel_message(int id, const char *data, int size);
    void set_state(LibdatachannelPeerState state);
    void report_error(const std::string &message);
    std::string channel_label(int id) const;

    LibdatachannelConfig config_;
    LibdatachannelCallbacks callbacks_;
    WebrtcTransport transport_;
    int peer_connection_ = -1;
    int reliable_channel_ = -1;
    int realtime_channel_ = -1;
    LibdatachannelPeerState state_ = LibdatachannelPeerState::New;
    bool started_ = false;
    bool closed_ = false;
    mutable std::recursive_mutex mutex_;
};

} // namespace rs::realtime

#endif // RS_ENABLE_LIBDATACHANNEL
