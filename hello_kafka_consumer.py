"""
Hello-world check: listen for messages on a Kafka topic.
Run this FIRST (it waits for messages), then run hello_kafka_producer.py.
This script will keep running -- press Ctrl+C to stop it once you see the message.
"""
from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'test-topic',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    value_deserializer=lambda v: json.loads(v.decode('utf-8'))
)

print("Listening for messages on 'test-topic'... (Ctrl+C to stop)")
for message in consumer:
    print(f"Received: {message.value}")