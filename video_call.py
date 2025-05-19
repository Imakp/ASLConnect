class VideoCallManager:
    """
    Manages video call rooms and participants.
    """
    
    def __init__(self):
        """Initialize the video call manager."""
        self.rooms = {}  # Dictionary to track rooms and users
        print("VideoCallManager initialized")
    
    def add_user_to_room(self, room_id, user_id):
        """
        Add a user to a room.
        
        Parameters:
        -----------
        room_id : str
            Room identifier
        user_id : str
            User identifier
        """
        if room_id not in self.rooms:
            self.rooms[room_id] = set()
        
        self.rooms[room_id].add(user_id)
        print(f"User {user_id} joined room {room_id}")
        print(f"Current rooms: {self.rooms}")
    
    def remove_user_from_room(self, room_id, user_id):
        """
        Remove a user from a room.
        
        Parameters:
        -----------
        room_id : str
            Room identifier
        user_id : str
            User identifier
        """
        if room_id in self.rooms and user_id in self.rooms[room_id]:
            self.rooms[room_id].remove(user_id)
            print(f"User {user_id} left room {room_id}")
            
            # Clean up empty rooms
            if not self.rooms[room_id]:
                del self.rooms[room_id]
                print(f"Room {room_id} is now empty and has been removed")
            
            print(f"Current rooms: {self.rooms}")
    
    def get_users_in_room(self, room_id):
        """
        Get all users in a room.
        
        Parameters:
        -----------
        room_id : str
            Room identifier
            
        Returns:
        --------
        users : set
            Set of user identifiers in the room
        """
        return self.rooms.get(room_id, set())
    
    def get_room_count(self):
        """
        Get the number of active rooms.
        
        Returns:
        --------
        count : int
            Number of active rooms
        """
        return len(self.rooms)
