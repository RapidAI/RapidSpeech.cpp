#pragma once

#include "rs_realtime_protocol.h"

#include <functional>
#include <mutex>
#include <vector>

namespace rs::realtime
{

// The bridge keeps the AOQ-like protocol independent from a specific WebRTC
// implementation. Its callbacks are connected to a libwebrtc or
// libdatachannel DataChannel by the integration layer.
struct WebrtcDataChannelBridge
{
    std::function<bool(const uint8_t *data, size_t size)> send_reliable;
    std::function<bool(const uint8_t *data, size_t size)> send_unreliable;
};

class WebrtcTransport final : public RealtimeTransport
{
public:
    explicit WebrtcTransport(WebrtcDataChannelBridge bridge);

    TransportKind kind() const override;
    bool connect() override;
    void close() override;
    bool send_datagram(const MediaObject &object) override;
    bool send_reliable(const MediaObject &object) override;
    TransportStats stats() const override;
    void set_observer(TransportObserver *observer) override;

    bool receive_unreliable(const uint8_t *data, size_t size);
    bool receive_reliable(const uint8_t *data, size_t size);
    bool connected() const;

private:
    bool send(const MediaObject &object, bool reliable);
    bool receive(const uint8_t *data, size_t size, bool reliable);
    void notify_error(const std::string &message);

    WebrtcDataChannelBridge bridge_;
    TransportObserver *observer_ = nullptr;
    TransportStats stats_;
    std::vector<uint8_t> reliable_receive_buffer_;
    bool connected_ = false;
    mutable std::mutex mutex_;
};

} // namespace rs::realtime
