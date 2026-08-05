#include "rs_webrtc_transport.h"

#include <cstddef>
#include <utility>

namespace rs::realtime
{
namespace
{

uint32_t read_be32(const uint8_t *data)
{
    return (static_cast<uint32_t>(data[0]) << 24) |
           (static_cast<uint32_t>(data[1]) << 16) |
           (static_cast<uint32_t>(data[2]) << 8) |
           static_cast<uint32_t>(data[3]);
}

} // namespace

WebrtcTransport::WebrtcTransport(WebrtcDataChannelBridge bridge)
    : bridge_(std::move(bridge))
{
}

TransportKind WebrtcTransport::kind() const
{
    return TransportKind::Webrtc;
}

bool WebrtcTransport::connect()
{
    std::lock_guard<std::mutex> lock(mutex_);
    connected_ = static_cast<bool>(bridge_.send_reliable) &&
                 static_cast<bool>(bridge_.send_unreliable);
    return connected_;
}

void WebrtcTransport::close()
{
    std::lock_guard<std::mutex> lock(mutex_);
    connected_ = false;
    reliable_receive_buffer_.clear();
}

bool WebrtcTransport::send_datagram(const MediaObject &object)
{
    return send(object, false);
}

bool WebrtcTransport::send_reliable(const MediaObject &object)
{
    return send(object, true);
}

TransportStats WebrtcTransport::stats() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void WebrtcTransport::set_observer(TransportObserver *observer)
{
    std::lock_guard<std::mutex> lock(mutex_);
    observer_ = observer;
}

bool WebrtcTransport::receive_unreliable(const uint8_t *data, size_t size)
{
    return receive(data, size, false);
}

bool WebrtcTransport::receive_reliable(const uint8_t *data, size_t size)
{
    return receive(data, size, true);
}

bool WebrtcTransport::connected() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return connected_;
}

bool WebrtcTransport::send(const MediaObject &object, bool reliable)
{
    bool connected = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        connected = connected_;
    }
    if (!connected) {
        notify_error("WebRTC transport is disconnected");
        return false;
    }

    const std::vector<uint8_t> encoded = encode_object(object);
    if (encoded.empty()) {
        notify_error("WebRTC object encoding failed");
        return false;
    }

    const auto &sender = reliable ? bridge_.send_reliable : bridge_.send_unreliable;
    if (!sender(encoded.data(), encoded.size())) return false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++stats_.packets_sent;
        stats_.bytes_sent += encoded.size();
    }
    return true;
}

bool WebrtcTransport::receive(const uint8_t *data, size_t size, bool reliable)
{
    if (!data || size == 0) return true;
    if (!reliable) {
        const DecodeResult decoded = decode_object(data, size);
        if (!decoded.ok) {
            notify_error(decoded.error);
            return false;
        }
        TransportObserver *observer = nullptr;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++stats_.packets_received;
            stats_.bytes_received += size;
            observer = observer_;
        }
        if (observer) observer->on_transport_object(decoded.object);
        return true;
    }

    std::vector<MediaObject> objects;
    std::string error;
    TransportObserver *observer = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        reliable_receive_buffer_.insert(reliable_receive_buffer_.end(), data, data + size);
        while (reliable_receive_buffer_.size() >= k_object_header_size) {
            const uint32_t payload_length = read_be32(reliable_receive_buffer_.data() + 49);
            const size_t object_size = k_object_header_size + payload_length;
            if (object_size > reliable_receive_buffer_.size()) break;
            const DecodeResult decoded = decode_object(reliable_receive_buffer_.data(), object_size);
            if (!decoded.ok) {
                error = decoded.error;
                reliable_receive_buffer_.clear();
                break;
            }
            reliable_receive_buffer_.erase(
                reliable_receive_buffer_.begin(),
                reliable_receive_buffer_.begin() + static_cast<std::ptrdiff_t>(object_size));
            ++stats_.packets_received;
            stats_.bytes_received += object_size;
            objects.push_back(decoded.object);
        }
        observer = observer_;
    }

    if (!error.empty()) {
        notify_error(error);
        return false;
    }
    if (observer) {
        for (auto &object : objects) observer->on_transport_object(std::move(object));
    }
    return true;
}

void WebrtcTransport::notify_error(const std::string &message)
{
    TransportObserver *observer = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        observer = observer_;
    }
    if (observer) observer->on_transport_error(message);
}

} // namespace rs::realtime
