#include "rs_libdatachannel_peer.h"

#ifdef RS_ENABLE_LIBDATACHANNEL

#include <rtc/rtc.h>

#include <climits>
#include <utility>

namespace rs::realtime
{
namespace
{

LibdatachannelPeerState map_state(rtcState state)
{
    switch (state) {
    case RTC_CONNECTING:
        return LibdatachannelPeerState::Connecting;
    case RTC_CONNECTED:
        return LibdatachannelPeerState::Connected;
    case RTC_DISCONNECTED:
        return LibdatachannelPeerState::Disconnected;
    case RTC_FAILED:
        return LibdatachannelPeerState::Failed;
    case RTC_CLOSED:
        return LibdatachannelPeerState::Closed;
    case RTC_NEW:
    default:
        return LibdatachannelPeerState::New;
    }
}

std::string c_string(const char *value)
{
    return value ? std::string(value) : std::string();
}

} // namespace

LibdatachannelPeer::LibdatachannelPeer(LibdatachannelConfig config,
                                       LibdatachannelCallbacks callbacks)
    : config_(std::move(config)),
      callbacks_(std::move(callbacks)),
      transport_({
          [this](const uint8_t *data, size_t size) {
              return send_channel(reliable_channel_, data, size);
          },
          [this](const uint8_t *data, size_t size) {
              return send_channel(realtime_channel_, data, size);
          },
      })
{
}

LibdatachannelPeer::~LibdatachannelPeer()
{
    close();
}

bool LibdatachannelPeer::start(bool offerer)
{
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (started_) return false;

    std::vector<const char *> ice_servers;
    ice_servers.reserve(config_.ice_servers.size());
    for (const auto &server : config_.ice_servers) {
        ice_servers.push_back(server.c_str());
    }

    rtcConfiguration config{};
    config.iceServers = ice_servers.empty() ? nullptr : ice_servers.data();
    config.iceServersCount = static_cast<int>(ice_servers.size());
    config.certificateType = RTC_CERTIFICATE_DEFAULT;
    config.iceTransportPolicy = RTC_TRANSPORT_POLICY_ALL;
    config.maxMessageSize = config_.max_message_size;
    config.disableAutoNegotiation = false;

    peer_connection_ = rtcCreatePeerConnection(&config);
    if (peer_connection_ < 0) {
        report_error("libdatachannel failed to create PeerConnection");
        return false;
    }

    rtcSetUserPointer(peer_connection_, this);
    rtcSetLocalDescriptionCallback(peer_connection_, &LibdatachannelPeer::on_local_description);
    rtcSetLocalCandidateCallback(peer_connection_, &LibdatachannelPeer::on_local_candidate);
    rtcSetStateChangeCallback(peer_connection_, &LibdatachannelPeer::on_state_change);
    rtcSetDataChannelCallback(peer_connection_, &LibdatachannelPeer::on_data_channel);

    started_ = true;
    closed_ = false;
    set_state(LibdatachannelPeerState::New);

    if (offerer && !create_protocol_channels()) {
        close();
        return false;
    }

    if (!transport_.connect()) {
        report_error("WebRTC transport callbacks are not configured");
        close();
        return false;
    }
    return true;
}

bool LibdatachannelPeer::set_remote_description(const std::string &sdp,
                                                const std::string &type)
{
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!started_ || closed_ || sdp.empty() || type.empty()) return false;
    return rtcSetRemoteDescription(peer_connection_, sdp.c_str(), type.c_str()) ==
           RTC_ERR_SUCCESS;
}

bool LibdatachannelPeer::add_remote_candidate(const std::string &candidate,
                                              const std::string &mid)
{
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!started_ || closed_ || candidate.empty()) return false;
    return rtcAddRemoteCandidate(peer_connection_, candidate.c_str(), mid.c_str()) ==
           RTC_ERR_SUCCESS;
}

void LibdatachannelPeer::close()
{
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (closed_) return;
    closed_ = true;
    started_ = false;
    transport_.close();

    const int realtime_channel = realtime_channel_;
    const int reliable_channel = reliable_channel_;
    const int peer_connection = peer_connection_;
    realtime_channel_ = -1;
    reliable_channel_ = -1;
    peer_connection_ = -1;

    if (realtime_channel >= 0) {
        rtcSetUserPointer(realtime_channel, nullptr);
        rtcSetClosedCallback(realtime_channel, nullptr);
        rtcSetErrorCallback(realtime_channel, nullptr);
        rtcSetMessageCallback(realtime_channel, nullptr);
        rtcClose(realtime_channel);
        rtcDeleteDataChannel(realtime_channel);
    }
    if (reliable_channel >= 0) {
        rtcSetUserPointer(reliable_channel, nullptr);
        rtcSetClosedCallback(reliable_channel, nullptr);
        rtcSetErrorCallback(reliable_channel, nullptr);
        rtcSetMessageCallback(reliable_channel, nullptr);
        rtcClose(reliable_channel);
        rtcDeleteDataChannel(reliable_channel);
    }
    if (peer_connection >= 0) {
        rtcSetUserPointer(peer_connection, nullptr);
        rtcSetLocalDescriptionCallback(peer_connection, nullptr);
        rtcSetLocalCandidateCallback(peer_connection, nullptr);
        rtcSetStateChangeCallback(peer_connection, nullptr);
        rtcSetDataChannelCallback(peer_connection, nullptr);
        rtcClosePeerConnection(peer_connection);
        rtcDeletePeerConnection(peer_connection);
    }

    set_state(LibdatachannelPeerState::Closed);
}

bool LibdatachannelPeer::started() const
{
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    return started_;
}

bool LibdatachannelPeer::connected() const
{
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    return state_ == LibdatachannelPeerState::Connected &&
           reliable_channel_ >= 0 && realtime_channel_ >= 0 &&
           rtcIsOpen(reliable_channel_) && rtcIsOpen(realtime_channel_);
}

LibdatachannelPeerState LibdatachannelPeer::state() const
{
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    return state_;
}

TransportStats LibdatachannelPeer::stats() const
{
    return transport_.stats();
}

WebrtcTransport &LibdatachannelPeer::transport()
{
    return transport_;
}

const WebrtcTransport &LibdatachannelPeer::transport() const
{
    return transport_;
}

void LibdatachannelPeer::on_local_description(int, const char *sdp, const char *type, void *ptr)
{
    auto *self = static_cast<LibdatachannelPeer *>(ptr);
    if (!self || !self->callbacks_.on_local_description) return;
    std::lock_guard<std::recursive_mutex> lock(self->mutex_);
    self->callbacks_.on_local_description(c_string(sdp), c_string(type));
}

void LibdatachannelPeer::on_local_candidate(int, const char *candidate, const char *mid, void *ptr)
{
    auto *self = static_cast<LibdatachannelPeer *>(ptr);
    if (!self || !self->callbacks_.on_local_candidate) return;
    std::lock_guard<std::recursive_mutex> lock(self->mutex_);
    self->callbacks_.on_local_candidate(c_string(candidate), c_string(mid));
}

void LibdatachannelPeer::on_state_change(int, rtcState state, void *ptr)
{
    auto *self = static_cast<LibdatachannelPeer *>(ptr);
    if (!self) return;
    std::lock_guard<std::recursive_mutex> lock(self->mutex_);
    self->set_state(map_state(state));
}

void LibdatachannelPeer::on_data_channel(int, int dc, void *ptr)
{
    auto *self = static_cast<LibdatachannelPeer *>(ptr);
    if (!self || dc < 0) return;
    std::lock_guard<std::recursive_mutex> lock(self->mutex_);

    const std::string label = self->channel_label(dc);
    if (label != self->config_.reliable_label && label != self->config_.realtime_label) {
        self->report_error("Ignoring unsupported libdatachannel DataChannel: " + label);
        rtcDeleteDataChannel(dc);
        return;
    }
    if (!self->bind_channel(dc, label)) {
        self->report_error("Duplicate or invalid protocol DataChannel: " + label);
        rtcDeleteDataChannel(dc);
    }
}

void LibdatachannelPeer::on_channel_closed(int id, void *ptr)
{
    auto *self = static_cast<LibdatachannelPeer *>(ptr);
    if (!self) return;
    std::lock_guard<std::recursive_mutex> lock(self->mutex_);
    if ((id == self->reliable_channel_ || id == self->realtime_channel_) &&
        self->state_ == LibdatachannelPeerState::Connected) {
        self->set_state(LibdatachannelPeerState::Disconnected);
    }
}

void LibdatachannelPeer::on_channel_error(int, const char *error, void *ptr)
{
    auto *self = static_cast<LibdatachannelPeer *>(ptr);
    if (self) self->report_error(c_string(error));
}

void LibdatachannelPeer::on_channel_message(int id, const char *data, int size, void *ptr)
{
    auto *self = static_cast<LibdatachannelPeer *>(ptr);
    if (self) self->handle_channel_message(id, data, size);
}

bool LibdatachannelPeer::create_protocol_channels()
{
    reliable_channel_ = rtcCreateDataChannel(peer_connection_, config_.reliable_label.c_str());
    if (reliable_channel_ < 0 || !bind_channel(reliable_channel_, config_.reliable_label)) {
        return false;
    }

    rtcDataChannelInit init{};
    init.reliability.unordered = true;
    init.reliability.unreliable = true;
    init.reliability.maxRetransmits = 0;
    init.protocol = "rs-object-v1";
    realtime_channel_ = rtcCreateDataChannelEx(peer_connection_, config_.realtime_label.c_str(),
                                                &init);
    if (realtime_channel_ < 0 || !bind_channel(realtime_channel_, config_.realtime_label)) {
        return false;
    }
    return true;
}

bool LibdatachannelPeer::bind_channel(int id, const std::string &label)
{
    if (id < 0) return false;
    int *slot = label == config_.reliable_label ? &reliable_channel_ :
                label == config_.realtime_label ? &realtime_channel_ : nullptr;
    if (!slot || (*slot >= 0 && *slot != id)) return false;

    *slot = id;
    rtcSetUserPointer(id, this);
    rtcSetClosedCallback(id, &LibdatachannelPeer::on_channel_closed);
    rtcSetErrorCallback(id, &LibdatachannelPeer::on_channel_error);
    rtcSetMessageCallback(id, &LibdatachannelPeer::on_channel_message);
    return true;
}

bool LibdatachannelPeer::send_channel(int id, const uint8_t *data, size_t size)
{
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!started_ || closed_ || id < 0 || !data || size > static_cast<size_t>(INT_MAX)) {
        return false;
    }
    if (!rtcIsOpen(id)) return false;
    return rtcSendMessage(id, reinterpret_cast<const char *>(data), static_cast<int>(size)) ==
           RTC_ERR_SUCCESS;
}

void LibdatachannelPeer::handle_channel_message(int id, const char *data, int size)
{
    if (!data || size < 0) {
        report_error("libdatachannel delivered a text message on a binary protocol channel");
        return;
    }

    bool realtime = false;
    bool reliable = false;
    {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        if (closed_) return;
        realtime = id == realtime_channel_;
        reliable = id == reliable_channel_;
    }
    if (realtime) {
        transport_.receive_unreliable(reinterpret_cast<const uint8_t *>(data),
                                      static_cast<size_t>(size));
    } else if (reliable) {
        transport_.receive_reliable(reinterpret_cast<const uint8_t *>(data),
                                     static_cast<size_t>(size));
    }
}

void LibdatachannelPeer::set_state(LibdatachannelPeerState state)
{
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    state_ = state;
    if (callbacks_.on_state_change) callbacks_.on_state_change(state);
}

void LibdatachannelPeer::report_error(const std::string &message)
{
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (callbacks_.on_error) callbacks_.on_error(message);
}

std::string LibdatachannelPeer::channel_label(int id) const
{
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    char buffer[256] = {};
    const int result = rtcGetDataChannelLabel(id, buffer, sizeof(buffer));
    if (result < 0) return {};
    return std::string(buffer);
}

} // namespace rs::realtime

#endif // RS_ENABLE_LIBDATACHANNEL
