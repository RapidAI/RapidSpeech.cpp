#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rs::realtime
{

constexpr uint8_t k_wire_version = 1;
constexpr size_t k_object_header_size = 53;

enum class TrackType : uint8_t
{
    Audio = 1,
    Video = 2,
    Data = 3,
};

enum class DeliveryMode : uint8_t
{
    Datagram = 0,
    ReliableStream = 1,
};

enum class ObjectKind : uint8_t
{
    AudioFrame = 1,
    VideoFrame = 2,
    DataMessage = 3,
};

enum class TransportKind : uint8_t
{
    Loopback = 0,
    Xquic = 1,
    Webrtc = 2,
};

struct TrackDescriptor
{
    uint32_t track_id = 0;
    TrackType type = TrackType::Data;
    ObjectKind object_kind = ObjectKind::DataMessage;
    DeliveryMode delivery = DeliveryMode::Datagram;
    uint8_t priority = 0;
    std::string codec;
};

struct ObjectHeader
{
    uint8_t version = k_wire_version;
    uint8_t flags = 0;
    TrackType track_type = TrackType::Data;
    DeliveryMode delivery = DeliveryMode::Datagram;
    ObjectKind object_kind = ObjectKind::DataMessage;
    uint32_t track_id = 0;
    uint64_t group_id = 0;
    uint64_t object_id = 0;
    uint64_t timestamp_us = 0;
    uint64_t deadline_us = 0;
    uint16_t fragment_index = 0;
    uint16_t fragment_count = 1;
    uint32_t payload_length = 0;
};

struct MediaObject
{
    ObjectHeader header;
    std::vector<uint8_t> payload;
};

struct DecodeResult
{
    bool ok = false;
    MediaObject object;
    std::string error;
};

std::vector<uint8_t> encode_object(const MediaObject &object);
DecodeResult decode_object(const uint8_t *data, size_t size);
uint8_t track_priority(TrackType type);

class RealtimeScheduler
{
public:
    explicit RealtimeScheduler(size_t max_queue_size = 256);

    bool enqueue(MediaObject object, uint64_t now_us);
    std::vector<MediaObject> drain(size_t max_objects, size_t max_bytes, uint64_t now_us);
    void clear();

    size_t size() const;
    uint64_t dropped_expired() const;
    uint64_t dropped_overflow() const;

private:
    struct QueuedObject
    {
        MediaObject object;
        uint8_t priority = 0;
        uint64_t enqueue_order = 0;
    };

    size_t max_queue_size_;
    uint64_t enqueue_order_ = 0;
    uint64_t dropped_expired_ = 0;
    uint64_t dropped_overflow_ = 0;
    std::vector<QueuedObject> queue_;
};

struct TransportStats
{
    uint64_t rtt_us = 0;
    uint64_t packets_lost = 0;
    uint64_t packets_sent = 0;
    uint64_t bytes_sent = 0;
    uint64_t packets_received = 0;
    uint64_t bytes_received = 0;
};

class TransportObserver
{
public:
    virtual ~TransportObserver() = default;
    virtual void on_transport_object(MediaObject object) = 0;
    virtual void on_transport_error(const std::string &message) = 0;
};

class RealtimeTransport
{
public:
    virtual ~RealtimeTransport() = default;

    virtual TransportKind kind() const = 0;
    virtual bool connect() = 0;
    virtual void close() = 0;
    virtual bool send_datagram(const MediaObject &object) = 0;
    virtual bool send_reliable(const MediaObject &object) = 0;
    virtual TransportStats stats() const = 0;
    virtual void set_observer(TransportObserver *observer) = 0;
};

class LoopbackTransport final : public RealtimeTransport
{
public:
    LoopbackTransport();

    TransportKind kind() const override;
    bool connect() override;
    void close() override;
    bool send_datagram(const MediaObject &object) override;
    bool send_reliable(const MediaObject &object) override;
    TransportStats stats() const override;
    void set_observer(TransportObserver *observer) override;

    bool connected() const;

private:
    bool send(const MediaObject &object);

    bool connected_ = false;
    TransportObserver *observer_ = nullptr;
    TransportStats stats_;
};

} // namespace rs::realtime
