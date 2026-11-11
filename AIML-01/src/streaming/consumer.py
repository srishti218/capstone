"""Kafka worker hosted by FastAPI, with at-least-once delivery."""
import json
import logging
import os

from fastapi import HTTPException
from kafka import KafkaConsumer, KafkaProducer, TopicPartition
from kafka.structs import OffsetAndMetadata
from pydantic import ValidationError

from src.api.schemas import Transaction


logger = logging.getLogger(__name__)
TRANSACTIONS = "transactions"
SCORES = "fraud-scores"
ERRORS = "fraud-errors"


def broker():
    return os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")


def publisher():
    return KafkaProducer(
        bootstrap_servers=broker(), acks="all", enable_idempotence=True,
        max_block_ms=10_000, request_timeout_ms=15_000,
        value_serializer=lambda value: json.dumps(value, allow_nan=False).encode(),
    )


def process_message(message, score, producer, consumer):
    key = message.key or f"{message.topic}:{message.partition}:{message.offset}".encode()
    try:
        transaction = Transaction.model_validate_json(message.value)
        result = score(transaction)
        topic, payload = SCORES, result.model_dump(exclude_none=True)
        key = transaction.transaction_id.encode()
    except ValidationError:
        topic, payload = ERRORS, {"error": "Invalid transaction schema"}
    except HTTPException as exc:
        if exc.status_code >= 500:
            raise
        topic, payload = ERRORS, {"error": exc.detail, "status": exc.status_code}

    if topic == ERRORS:
        payload.update(partition=message.partition, offset=message.offset)
    producer.send(topic, key=key, value=payload).get(timeout=20)
    consumer.commit({TopicPartition(message.topic, message.partition): OffsetAndMetadata(message.offset + 1, "")})


def consume(score, stop):
    while not stop.is_set():
        consumer = producer = None
        try:
            producer = publisher()
            consumer = KafkaConsumer(
                TRANSACTIONS, bootstrap_servers=broker(),
                group_id=os.getenv("KAFKA_GROUP_ID", "fraud-fastapi-v1"),
                auto_offset_reset="earliest", enable_auto_commit=False,
            )
            logger.info("Kafka scoring worker ready")
            while not stop.is_set():
                for records in consumer.poll(timeout_ms=1000, max_records=20).values():
                    for message in records:
                        if stop.is_set():
                            return
                        process_message(message, score, producer, consumer)
        except Exception:
            logger.exception("Kafka scoring worker will retry uncommitted messages")
            stop.wait(3)
        finally:
            if consumer is not None:
                consumer.close(autocommit=False)
            if producer is not None:
                producer.close(timeout=3)


if __name__ == "__main__":
    import uvicorn

    os.environ["KAFKA_ENABLED"] = "true"
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000)
