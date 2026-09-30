"""
Hello-world check: confirm Python can talk to Redis running in Docker.
Run this AFTER `docker-compose up -d` and BEFORE writing any real logic.
"""
import redis

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

r.set('test_key', 'hello from python')
value = r.get('test_key')

print(f"Wrote and read back: {value}")
print("Redis connection: OK" if value == 'hello from python' else "Redis connection: FAILED")