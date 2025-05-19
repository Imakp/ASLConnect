class WebRTCHandler {
    constructor(socket) {
        this.socket = socket;
        this.peerConnection = null;
        this.localStream = null;
        this.remoteStream = null;
        this.roomId = null;
        this.userId = socket.id; // Store the socket ID as user ID
        
        // WebRTC configuration
        this.config = {
            iceServers: [
                { urls: 'stun:stun.l.google.com:19302' }
            ]
        };
        
        this.setupSocketListeners();
    }
    
    setupSocketListeners() {
        this.socket.on('offer', async (data) => {
            await this.handleOffer(data);
        });
        
        this.socket.on('answer', async (data) => {
            await this.handleAnswer(data);
        });
        
        this.socket.on('ice_candidate', async (data) => {
            await this.handleIceCandidate(data);
        });
        
        // Store socket ID when connected
        this.socket.on('connect', () => {
            this.userId = this.socket.id;
            console.log('Connected with user ID:', this.userId);
        });
    }
    
    async initializeLocalStream() {
        try {
            this.localStream = await navigator.mediaDevices.getUserMedia({
                video: true,
                audio: true
            });
            const localVideo = document.getElementById('localVideo');
            localVideo.srcObject = this.localStream;
            
            // Add user ID as data attribute to the video element
            localVideo.setAttribute('data-user-id', this.userId);
            
            return true;
        } catch (error) {
            console.error('Error accessing media devices:', error);
            return false;
        }
    }
    
    async joinRoom(roomId) {
        this.roomId = roomId;
        await this.createPeerConnection();
        this.socket.emit('join', { room: roomId });
    }
    
    async createPeerConnection() {
        this.peerConnection = new RTCPeerConnection(this.config);
        
        // Add local stream tracks to peer connection
        this.localStream.getTracks().forEach(track => {
            this.peerConnection.addTrack(track, this.localStream);
        });
        
        // Handle incoming streams
        this.peerConnection.ontrack = (event) => {
            this.remoteStream = event.streams[0];
            const remoteVideo = document.getElementById('remoteVideo');
            remoteVideo.srcObject = this.remoteStream;
            
            // Add remote user ID attribute when we know it
            // This will be updated when we receive the first subtitle
        };
        
        // Handle ICE candidates
        this.peerConnection.onicecandidate = (event) => {
            if (event.candidate) {
                this.socket.emit('ice_candidate', {
                    room: this.roomId,
                    candidate: event.candidate
                });
            }
        };
        
        // Create and send offer
        try {
            const offer = await this.peerConnection.createOffer();
            await this.peerConnection.setLocalDescription(offer);
            this.socket.emit('offer', {
                room: this.roomId,
                offer: offer
            });
        } catch (error) {
            console.error('Error creating offer:', error);
        }
    }
    
    async handleOffer(data) {
        if (!this.peerConnection) {
            await this.createPeerConnection();
        }
        
        await this.peerConnection.setRemoteDescription(new RTCSessionDescription(data.offer));
        
        const answer = await this.peerConnection.createAnswer();
        await this.peerConnection.setLocalDescription(answer);
        
        this.socket.emit('answer', {
            room: this.roomId,
            answer: answer
        });
    }
    
    async handleAnswer(data) {
        await this.peerConnection.setRemoteDescription(new RTCSessionDescription(data.answer));
    }
    
    async handleIceCandidate(data) {
        if (data.candidate) {
            try {
                await this.peerConnection.addIceCandidate(new RTCIceCandidate(data.candidate));
            } catch (error) {
                console.error('Error adding ICE candidate:', error);
            }
        }
    }
    
    leaveRoom() {
        if (this.roomId) {
            this.socket.emit('leave', { room: this.roomId });
            this.cleanup();
        }
    }
    
    cleanup() {
        if (this.peerConnection) {
            this.peerConnection.close();
            this.peerConnection = null;
        }
        
        if (this.localStream) {
            this.localStream.getTracks().forEach(track => track.stop());
            this.localStream = null;
        }
        
        document.getElementById('localVideo').srcObject = null;
        document.getElementById('remoteVideo').srcObject = null;
        this.roomId = null;
    }
    
    toggleMute() {
        if (this.localStream) {
            const audioTrack = this.localStream.getAudioTracks()[0];
            if (audioTrack) {
                audioTrack.enabled = !audioTrack.enabled;
                return audioTrack.enabled;
            }
        }
        return false;
    }
}
