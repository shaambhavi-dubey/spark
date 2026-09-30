"""
Hello-world check: publish a single message to a Kafka topic.
Run this AFTER `docker-compose up -d`.
Run hello_kafka_consumer.py in a SEPARATE terminal at the same time (or after)
to see the message arrive.
"""
from kafka import KafkaProducer
import json
import time

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

message = {"hello": "from producer", "timestamp": time.time()}
producer.send('test-topic', message)
producer.flush()

print(f"Sent: {message}")