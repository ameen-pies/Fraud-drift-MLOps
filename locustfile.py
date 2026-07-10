"""Load testing with Locust — hits /predict with realistic payloads."""

import random
from locust import HttpUser, task, between


FEATURES = [
    "transaction_amount",
    "transaction_hour",
    "transaction_day",
    "merchant_category",
    "distance_from_home",
    "distance_from_last_txn",
    "ratio_to_median_amount",
    "is_international",
    "num_txns_last_1h",
    "num_txns_last_24h",
    "avg_amount_last_7d",
    "account_age_days",
    "num_chargebacks",
    "card_present",
    "pin_used",
]


def random_payload() -> dict:
    return {
        "transaction_amount": round(random.uniform(1, 2000), 2),
        "transaction_hour": random.randint(0, 23),
        "transaction_day": random.randint(0, 6),
        "merchant_category": round(random.uniform(0, 10), 2),
        "distance_from_home": round(random.uniform(0, 100), 2),
        "distance_from_last_txn": round(random.uniform(0, 50), 2),
        "ratio_to_median_amount": round(random.uniform(0.1, 5), 2),
        "is_international": random.choice([0, 1]),
        "num_txns_last_1h": round(random.uniform(0, 10), 2),
        "num_txns_last_24h": round(random.uniform(0, 20), 2),
        "avg_amount_last_7d": round(random.uniform(10, 1000), 2),
        "account_age_days": random.randint(30, 3650),
        "num_chargebacks": random.randint(0, 5),
        "card_present": random.choice([0, 1]),
        "pin_used": random.choice([0, 1]),
    }


class FraudAPIUser(HttpUser):
    wait_time = between(0.01, 0.1)
    host = "http://localhost:8000"

    @task
    def predict(self):
        self.client.post("/predict", json=random_payload())

    @task(2)
    def health(self):
        self.client.get("/health")
