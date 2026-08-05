#include "rs_realtime_protocol.h"

#include <algorithm>
#include <array>
#include <limits>
#include <utility>

namespace rs::realtime
{
namespace
{

constexpr std::array<uint8_t, 4> k_magic = {'R', 'S', 'O', 'Q'};

void append_u8(std::vector<uint8_t> &out, uint8_t value)
{
    out.push_back(value);
}

void append_u16(std::vector<uint8_t> &out, uint16_t value)
{
    out.push_back(static_cast<uint8_t>((value >> 8) & 0xff));
    out.push_back(static_cast<uint8_t>(value & 0xff));
}

void append_u32(std::vector<uint8_t> &out, uint32_t value)
{
    for (int shift = 24; shift >= 0; shift -= 8) {
        out.push_back(static_cast<uint8_t>((value >> shift) & 0xff));
    }
}

void append_u64(std::vector<uint8_t> &out, uint64_t value)
{
    for (int shift = 56; shift >= 0; shift -= 8) {
        out.push_back(static_cast<uint8_t>((value >> shift) & 0xff));
    }
}

bool read_u8(const uint8_t *data, size_t size, size_t &offset, uint8_t &value)
{
    if (offset >= size) return false;
    value = data[offset++];
    return true;
}

bool read_u16(const uint8_t *data, size_t size, size_t &offset, uint16_t &value)
{
    if (size - offset < 2) return false;
    value = static_cast<uint16_t>(data[offset] << 8 | data[offset + 1]);
    offset += 2;
    return true;
}

bool read_u32(const uint8_t *data, size_t size, size_t &offset, uint32_t &value)
{
    if (size - offset < 4) return false;
    value = (static_cast<uint32_t>(data[offset]) << 24) |
            (static_cast<uint32_t>(data[offset + 1]) << 16) |
            (static_cast<uint32_t>(data[offset + 2]) << 8) |
            static_cast<uint32_t>(data[offset + 3]);
    offset += 4;
    return true;
}

bool read_u64(const uint8_t *data, size_t size, size_t &offset, uint64_t &value)
{
    if (size - offset < 8) return false;
    value = 0;
    for (int i = 0; i < 8; ++i) {
        value = (value << 8) | data[offset + static_cast<size_t>(i)];
    }
    offset += 8;
    return true;
}

bool valid_track_type(uint8_t value)
{
    return value >= static_cast<uint8_t>(TrackType::Audio) &&
           value <= static_cast<uint8_t>(TrackType::Data);
}

bool valid_delivery_mode(uint8_t value)
{
    return value <= static_cast<uint8_t>(DeliveryMode::ReliableStream);
}

bool valid_object_kind(uint8_t value)
{
    return value >= static_cast<uint8_t>(ObjectKind::AudioFrame) &&
           value <= static_cast<uint8_t>(ObjectKind::DataMessage);
}

} // namespace

std::vector<uint8_t> encode_object(const MediaObject &object)
{
    const ObjectHeader &header = object.header;
    if (header.version != k_wire_version || header.fragment_count == 0 ||
        header.fragment_index >= header.fragment_count ||
        object.payload.size() > std::numeric_limits<uint32_t>::max()) {
        return {};
    }

    std::vector<uint8_t> encoded;
    encoded.reserve(k_object_header_size + object.payload.size());
    encoded.insert(encoded.end(), k_magic.begin(), k_magic.end());
    append_u8(encoded, header.version);
    append_u8(encoded, header.flags);
    append_u8(encoded, static_cast<uint8_t>(header.track_type));
    append_u8(encoded, static_cast<uint8_t>(header.delivery));
    append_u8(encoded, static_cast<uint8_t>(header.object_kind));
    append_u32(encoded, header.track_id);
    append_u64(encoded, header.group_id);
    append_u64(encoded, header.object_id);
    append_u64(encoded, header.timestamp_us);
    append_u64(encoded, header.deadline_us);
    append_u16(encoded, header.fragment_index);
    append_u16(encoded, header.fragment_count);
    append_u32(encoded, static_cast<uint32_t>(object.payload.size()));
    encoded.insert(encoded.end(), object.payload.begin(), object.payload.end());
    return encoded;
}

DecodeResult decode_object(const uint8_t *data, size_t size)
{
    DecodeResult result;
    if (!data || size < k_object_header_size) {
        result.error = "object is shorter than its header";
        return result;
    }

    for (size_t i = 0; i < k_magic.size(); ++i) {
        if (data[i] != k_magic[i]) {
            result.error = "invalid object magic";
            return result;
        }
    }

    size_t offset = k_magic.size();
    uint8_t version = 0;
    uint8_t flags = 0;
    uint8_t track_type = 0;
    uint8_t delivery = 0;
    uint8_t object_kind = 0;
    uint32_t track_id = 0;
    uint64_t group_id = 0;
    uint64_t object_id = 0;
    uint64_t timestamp_us = 0;
    uint64_t deadline_us = 0;
    uint16_t fragment_index = 0;
    uint16_t fragment_count = 0;
    uint32_t payload_length = 0;

    if (!read_u8(data, size, offset, version) ||
        !read_u8(data, size, offset, flags) ||
        !read_u8(data, size, offset, track_type) ||
        !read_u8(data, size, offset, delivery) ||
        !read_u8(data, size, offset, object_kind) ||
        !read_u32(data, size, offset, track_id) ||
        !read_u64(data, size, offset, group_id) ||
        !read_u64(data, size, offset, object_id) ||
        !read_u64(data, size, offset, timestamp_us) ||
        !read_u64(data, size, offset, deadline_us) ||
        !read_u16(data, size, offset, fragment_index) ||
        !read_u16(data, size, offset, fragment_count) ||
        !read_u32(data, size, offset, payload_length)) {
        result.error = "truncated object header";
        return result;
    }

    if (version != k_wire_version) {
        result.error = "unsupported object version";
        return result;
    }
    if (!valid_track_type(track_type) || !valid_delivery_mode(delivery) ||
        !valid_object_kind(object_kind)) {
        result.error = "invalid object enum";
        return result;
    }
    if (fragment_count == 0 || fragment_index >= fragment_count) {
        result.error = "invalid object fragment range";
        return result;
    }
    if (payload_length != size - offset) {
        result.error = "object payload length does not match datagram size";
        return result;
    }

    result.object.header.version = version;
    result.object.header.flags = flags;
    result.object.header.track_type = static_cast<TrackType>(track_type);
    result.object.header.delivery = static_cast<DeliveryMode>(delivery);
    result.object.header.object_kind = static_cast<ObjectKind>(object_kind);
    result.object.header.track_id = track_id;
    result.object.header.group_id = group_id;
    result.object.header.object_id = object_id;
    result.object.header.timestamp_us = timestamp_us;
    result.object.header.deadline_us = deadline_us;
    result.object.header.fragment_index = fragment_index;
    result.object.header.fragment_count = fragment_count;
    result.object.header.payload_length = payload_length;
    result.object.payload.assign(data + offset, data + size);
    result.ok = true;
    return result;
}

RealtimeScheduler::RealtimeScheduler(size_t max_queue_size)
    : max_queue_size_(std::max<size_t>(1, max_queue_size))
{
}

bool RealtimeScheduler::enqueue(MediaObject object, uint64_t now_us)
{
    if (object.header.deadline_us != 0 && object.header.deadline_us <= now_us) {
        ++dropped_expired_;
        return false;
    }

    if (queue_.size() >= max_queue_size_) {
        auto worst = std::min_element(
            queue_.begin(), queue_.end(), [](const QueuedObject &a, const QueuedObject &b) {
                if (a.priority != b.priority) return a.priority < b.priority;
                return a.enqueue_order < b.enqueue_order;
            });
        const uint8_t priority = track_priority(object.header.track_type);
        if (worst == queue_.end() || worst->priority > priority) {
            ++dropped_overflow_;
            return false;
        }
        queue_.erase(worst);
        ++dropped_overflow_;
    }

    const uint8_t priority = track_priority(object.header.track_type);
    queue_.push_back({std::move(object), priority, enqueue_order_++});
    return true;
}

std::vector<MediaObject> RealtimeScheduler::drain(size_t max_objects, size_t max_bytes, uint64_t now_us)
{
    std::vector<MediaObject> output;
    output.reserve(std::min(max_objects, queue_.size()));
    size_t bytes = 0;

    queue_.erase(std::remove_if(queue_.begin(), queue_.end(), [&](const QueuedObject &queued) {
        if (queued.object.header.deadline_us != 0 &&
            queued.object.header.deadline_us <= now_us) {
            ++dropped_expired_;
            return true;
        }
        return false;
    }), queue_.end());

    std::stable_sort(queue_.begin(), queue_.end(), [](const QueuedObject &a, const QueuedObject &b) {
        if (a.priority != b.priority) return a.priority > b.priority;
        if (a.object.header.deadline_us != b.object.header.deadline_us) {
            if (a.object.header.deadline_us == 0) return false;
            if (b.object.header.deadline_us == 0) return true;
            return a.object.header.deadline_us < b.object.header.deadline_us;
        }
        return a.enqueue_order < b.enqueue_order;
    });

    while (!queue_.empty() && output.size() < max_objects) {
        const size_t object_bytes = queue_.front().object.payload.size();
        if (object_bytes > max_bytes - std::min(bytes, max_bytes)) break;
        bytes += object_bytes;
        output.push_back(std::move(queue_.front().object));
        queue_.erase(queue_.begin());
    }
    return output;
}

void RealtimeScheduler::clear()
{
    queue_.clear();
}

size_t RealtimeScheduler::size() const
{
    return queue_.size();
}

uint64_t RealtimeScheduler::dropped_expired() const
{
    return dropped_expired_;
}

uint64_t RealtimeScheduler::dropped_overflow() const
{
    return dropped_overflow_;
}

TransportKind LoopbackTransport::kind() const
{
    return TransportKind::Loopback;
}

LoopbackTransport::LoopbackTransport() = default;

bool LoopbackTransport::connect()
{
    connected_ = true;
    return true;
}

void LoopbackTransport::close()
{
    connected_ = false;
}

bool LoopbackTransport::send_datagram(const MediaObject &object)
{
    return send(object);
}

bool LoopbackTransport::send_reliable(const MediaObject &object)
{
    return send(object);
}

TransportStats LoopbackTransport::stats() const
{
    return stats_;
}

void LoopbackTransport::set_observer(TransportObserver *observer)
{
    observer_ = observer;
}

bool LoopbackTransport::connected() const
{
    return connected_;
}

bool LoopbackTransport::send(const MediaObject &object)
{
    if (!connected_) {
        if (observer_) observer_->on_transport_error("loopback transport is disconnected");
        return false;
    }

    const std::vector<uint8_t> encoded = encode_object(object);
    if (encoded.empty()) {
        if (observer_) observer_->on_transport_error("object encoding failed");
        return false;
    }

    ++stats_.packets_sent;
    stats_.bytes_sent += encoded.size();
    ++stats_.packets_received;
    stats_.bytes_received += encoded.size();
    if (observer_) {
        const DecodeResult decoded = decode_object(encoded.data(), encoded.size());
        if (!decoded.ok) {
            observer_->on_transport_error(decoded.error);
            return false;
        }
        observer_->on_transport_object(decoded.object);
    }
    return true;
}

uint8_t track_priority(TrackType type)
{
    switch (type) {
    case TrackType::Audio: return 3;
    case TrackType::Video: return 2;
    case TrackType::Data: return 1;
    }
    return 0;
}

} // namespace rs::realtime
