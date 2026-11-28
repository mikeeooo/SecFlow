"""The simulated live stream: one producer task in the backend submits MIA payments to `simulate`.

Browser tabs only read the feed and call start, stop or mode, so extra tabs never add submissions.
The generator only builds payment inputs (MDL, a scheme that fits the payee, within the sender's
per-transaction limit) from observed history plus a few unusual patterns. The decision always comes
from the assessment, never from the generator.
"""

from __future__ import annotations

import asyncio
import logging
import math
import uuid
from collections import deque
from typing import Any

import numpy as np

from backend.schemas import SimulationRequest
from backend.service import SecFlowService, utc_now_iso
from datagen.identities import BILLER_CATEGORIES, FIRST_NAMES, LAST_NAMES, QR_CATEGORIES

log = logging.getLogger("secflow.stream")
BATCH_SIZES = {"continuous": 1, "batch_5": 5, "batch_10": 10, "batch_20": 20}
FOREIGN = ("NL", "US", "DE", "GB", "SG")


class InputGenerator:
    """Plausible payment inputs built from what the service has seen so far."""

    def __init__(self, service: SecFlowService, seed: int = 7) -> None:
        self.service = service
        self.rng = np.random.default_rng(seed)
        history = service.history.sender
        self.senders = [a for a in service.accounts if a in history]
        weights = np.array([len(history[a].keys) for a in self.senders], dtype=float)
        self.weights = weights / weights.sum()
        self.queue: deque[dict[str, Any]] = deque()  # the rest of a burst

    def pick(self, options: list[Any] | tuple[Any, ...], p: list[float] | None = None) -> Any:
        return options[int(self.rng.choice(len(options), p=p))]

    def amount(self, sender: str, low: float, high: float) -> float:
        """A multiple of the sender's typical amount, kept under the MIA per-transaction limit."""
        series = self.service.history.sender.get(sender)
        recent = sorted(series.amounts[-25:]) if series else [40.0]
        value = recent[len(recent) // 2] * float(self.rng.uniform(low, high)) * math.exp(self.rng.normal(0, 0.2))
        cap = self.service.limits.for_segment(self.service.accounts[sender].account_type).per_transaction * 0.999
        return round(min(max(1.0, value), cap), 2)

    def scheme(self, sender: str, recipient: str) -> str:
        """An MIA scheme that fits the sender and the payee, as the dataset generator chooses it."""
        account = self.service.accounts.get(recipient)
        counterparty = self.service.counterparties.get(recipient)
        kind = ("business" if account.account_type == "business" else "person") if account else (
            counterparty.kind if counterparty else "person")
        category = counterparty.category if counterparty else "individual"
        if kind == "public_institution":
            return "p2g"
        if self.service.accounts[sender].account_type == "business":
            return "p2p_iban" if kind == "person" else "b2b"
        if kind == "business" and category in BILLER_CATEGORIES:
            return self.pick(("request_to_pay", "p2b_link", "p2b_qr_dynamic"), [0.25, 0.45, 0.30])
        if kind == "business" and category in QR_CATEGORIES:
            return self.pick(("p2b_qr_static", "p2b_qr_dynamic", "p2b_link"), [0.55, 0.30, 0.15])
        if kind == "business":
            return "p2b_link"
        return self.pick(("p2p_alias", "p2p_iban"), [0.83, 0.17])

    def next_payload(self) -> tuple[SimulationRequest, str | None]:
        """The next input and, for burst items, a token that ties them to one new beneficiary."""
        spec = self.queue.popleft() if self.queue else self.fresh()
        token = spec.pop("_burst", None)
        return SimulationRequest(request_id=f"LIVE-{uuid.uuid4().hex[:12]}", **spec), token

    def fresh(self) -> dict[str, Any]:
        sender = self.pick(self.senders, list(self.weights))
        home = self.service.accounts[sender].home_country
        spec: dict[str, Any] = {
            "sender_id": sender, "device_id": self.service.last_device(sender), "ip_country": home,
            "auth_method": self.pick(("biometric", "biometric", "password", "password_otp")),
            "failed_login_attempts": int(self.rng.random() < 0.03), "channel": "mobile_app"}
        series = self.service.history.sender.get(sender)
        recipients = [r for r in (series.recipients[-30:] if series else []) if r != sender] or ["CP-00001"]
        known = self.pick(recipients)
        new_person = {"new_recipient_name": f"{self.pick(FIRST_NAMES)} {self.pick(LAST_NAMES)}",
                      "payment_scheme": "p2p_iban" if self.service.accounts[sender].account_type == "business"
                      else "p2p_alias"}
        roll = float(self.rng.random())
        if roll < 0.80:  # an ordinary payment to a known recipient
            return {**spec, "recipient_id": known, "payment_scheme": self.scheme(sender, known),
                    "amount": self.amount(sender, 0.7, 1.3)}
        if roll < 0.86:  # an ordinary amount to someone new
            return {**spec, **new_person, "amount": self.amount(sender, 0.8, 2.0)}
        if roll < 0.90:  # a large planned purchase with strong sign-in
            return {**spec, "recipient_id": known, "payment_scheme": self.scheme(sender, known),
                    "amount": self.amount(sender, 6, 14), "auth_method": "password_otp"}
        if roll < 0.93:  # phishing-like takeover: failed sign-ins, a new device, near the cap
            return {**spec, **new_person, "device_id": None, "amount": self.amount(sender, 30, 60),
                    "failed_login_attempts": int(self.rng.integers(2, 5)), "auth_method": "password",
                    "ip_country": self.pick(FOREIGN) if self.rng.random() < 0.5 else home}
        if roll < 0.95 and new_person["payment_scheme"] == "p2p_alias":  # Request to Pay from a stranger
            return {**spec, **new_person, "payment_scheme": "request_to_pay", "amount": self.amount(sender, 2, 6)}
        if roll < 0.97:  # a rapid burst to one new beneficiary; later items pay the one the first created
            token = uuid.uuid4().hex
            burst = [{**spec, **new_person, "amount": self.amount(sender, 0.8, 1.6), "_burst": token}
                     for _ in range(int(self.rng.integers(4, 7)))]
            self.queue.extend(burst[1:])
            return burst[0]
        flagged = self.service.simulator_options(None)["flagged_beneficiaries"]
        target = self.pick(flagged)["id"] if flagged else known  # a beneficiary with active trusted evidence
        return {**spec, "recipient_id": target, "payment_scheme": self.scheme(sender, target),
                "amount": self.amount(sender, 0.6, 1.4)}


class StreamController:
    def __init__(self, service: SecFlowService, interval: float, batch_interval: float) -> None:
        self.service = service
        self.generator = InputGenerator(service)
        self.interval, self.batch_interval = interval, batch_interval
        self.mode = "continuous"
        self.running = False
        self.produced = self.failures = 0
        self.last_error: str | None = None
        self.started_at: str | None = None
        self.last_tick_at: str | None = None
        self._task: asyncio.Task[None] | None = None
        self._wake = asyncio.Event()
        self._burst_recipients: dict[str, str] = {}

    def status(self) -> dict[str, Any]:
        return {"running": self.running, "mode": self.mode,
                "interval_seconds": self.interval if self.mode == "continuous" else self.batch_interval,
                "produced": self.produced, "failures": self.failures, "last_error": self.last_error,
                "started_at": self.started_at, "last_tick_at": self.last_tick_at, "owner": "backend"}

    def start(self) -> dict[str, Any]:
        if not self.running:
            self.running, self.started_at = True, utc_now_iso()
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop(), name="secflow-stream")
        self._wake.set()
        return self.status()

    def stop(self) -> dict[str, Any]:
        self.running = False  # checked before every submission
        return self.status()

    def set_mode(self, mode: str) -> dict[str, Any]:
        self.mode = mode
        self._wake.set()
        return self.status()

    async def shutdown(self) -> None:
        self.running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def tick(self) -> int:
        """Submit one tick of simulated payments; returns how many were recorded."""
        committed = 0
        for _ in range(BATCH_SIZES[self.mode]):
            if not self.running:
                break
            payload, token = self.generator.next_payload()
            if token in self._burst_recipients:
                payload = payload.model_copy(update={"recipient_id": self._burst_recipients[token],
                                                     "new_recipient_name": None})
            try:
                detail, _ = await asyncio.to_thread(self.service.simulate, payload, "stream")
            except Exception as error:  # noqa: BLE001 - keep producing; the status shows the error
                self.failures += 1
                self.last_error = f"{type(error).__name__}: {error}"
                log.warning("Stream submission failed: %s", self.last_error)
                continue
            if token is not None:
                self._burst_recipients.setdefault(token, detail.recipient.id)
            self.produced += 1
            committed += 1
        self.last_tick_at = utc_now_iso()
        return committed

    async def _loop(self) -> None:
        while True:
            if self.running:
                await self.tick()
            delay = (self.interval if self.mode == "continuous" else self.batch_interval) if self.running else 3600.0
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=delay)
            except TimeoutError:
                pass
