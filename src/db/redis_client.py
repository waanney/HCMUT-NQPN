import json
import time
import redis

class RedisMemory:
    def __init__(self, host="localhost", port=6379, db=0, ttl_seconds=600):
        """
        Short-term memory storage using Redis.
        - ttl_seconds: how long each session stays alive in memory (default: 10 minutes)
        """
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.ttl = ttl_seconds

    def _key(self, session_id: str) -> str:
        """Builds a Redis key name for the given session."""
        return f"chat_memory:{session_id}"

    def add_message(self, session_id: str, role: str, content: str):
        """
        Adds a single message to the short-term memory.
        Automatically resets TTL on each new message.
        """
        key = self._key(session_id)
        message = {"role": role, "content": content, "timestamp": time.time()}

        # Push message to the right end of the Redis list
        self.client.rpush(key, json.dumps(message))
        # Refresh expiration timer for this chat session
        self.client.expire(key, self.ttl)

    def get_history(self, session_id: str, limit: int = 10):
        """
        Retrieves the last N messages from memory.
        Default limit = 10 (most recent messages).
        """
        key = self._key(session_id)
        items = self.client.lrange(key, -limit, -1)
        return [json.loads(i) for i in items]

    def clear(self, session_id: str):
        """Deletes all stored messages for the given session."""
        self.client.delete(self._key(session_id))
