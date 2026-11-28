"""Stable generated identities for SecFlow accounts and external MIA participants.

Every identity here is generated. Names are plausible Moldovan-style combinations of common first
and last names, and business names combine generic place words with a trade; none refers to a
verified real person, business or public body. Phone aliases follow the MIA format (`+373` and
eight digits) but are drawn at random from a seeded generator, so they are not real numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

IDENTITY_SEED = 42
ACCOUNT_COUNT = 400
HOME_COUNTRY = "MD"

FIRST_NAMES = (
    "Ion",
    "Maria",
    "Andrei",
    "Elena",
    "Mihai",
    "Ana",
    "Vasile",
    "Natalia",
    "Gheorghe",
    "Tatiana",
    "Sergiu",
    "Olga",
    "Victor",
    "Irina",
    "Alexandru",
    "Cristina",
    "Dumitru",
    "Svetlana",
    "Nicolae",
    "Ludmila",
    "Vladimir",
    "Galina",
    "Igor",
    "Valentina",
    "Dorin",
    "Aurelia",
    "Vitalie",
    "Doina",
    "Eugen",
    "Lilia",
    "Radu",
    "Corina",
    "Marin",
    "Daniela",
    "Petru",
    "Mariana",
    "Valeriu",
    "Angela",
    "Iurie",
    "Oxana",
    "Pavel",
    "Ecaterina",
    "Artur",
    "Viorica",
    "Denis",
    "Nadejda",
    "Stanislav",
    "Rodica",
    "Constantin",
    "Inga",
    "Octavian",
    "Larisa",
    "Grigore",
    "Silvia",
    "Veaceslav",
    "Victoria",
    "Anatol",
    "Alina",
    "Calin",
    "Diana",
    "Roman",
    "Iulia",
    "Ilie",
    "Carolina",
    "Tudor",
    "Snejana",
    "Valentin",
    "Adriana",
    "Emil",
    "Veronica",
    "Cornel",
    "Liliana",
    "Ruslan",
    "Marcela",
    "Vadim",
    "Stela",
    "Dan",
    "Zinaida",
    "Maxim",
    "Ina",
)

LAST_NAMES = (
    "Rusu",
    "Ceban",
    "Ciobanu",
    "Popa",
    "Lungu",
    "Munteanu",
    "Cojocaru",
    "Rotaru",
    "Sirbu",
    "Turcan",
    "Botnari",
    "Moraru",
    "Bivol",
    "Gutu",
    "Melnic",
    "Railean",
    "Cebotari",
    "Bodiu",
    "Grosu",
    "Croitoru",
    "Postica",
    "Ticu",
    "Pascari",
    "Stratan",
    "Ursu",
    "Birca",
    "Chiriac",
    "Plamadeala",
    "Vrabie",
    "Rata",
    "Golban",
    "Lupu",
    "Covalciuc",
    "Bejan",
    "Girlea",
    "Mocanu",
    "Pirlog",
    "Andronic",
    "Doroftei",
    "Taranu",
    "Negru",
    "Prisacaru",
    "Spinu",
    "Bulat",
    "Ribac",
    "Cazacu",
    "Ababii",
    "Codreanu",
    "Graur",
    "Scutaru",
    "Volosciuc",
    "Tabac",
    "Bostan",
    "Caraman",
    "Donos",
    "Esanu",
    "Furtuna",
    "Gaina",
    "Hincu",
    "Iachim",
    "Jitaru",
    "Leahu",
    "Malai",
    "Nastas",
    "Olaru",
    "Palii",
    "Rosca",
    "Sava",
    "Toma",
    "Ungureanu",
    "Vieru",
    "Zaharia",
    "Arama",
    "Bargan",
    "Chirtoaca",
    "Dima",
    "Enache",
    "Frunza",
    "Gorincioi",
    "Harea",
)

BUSINESS_PREFIXES = (
    "Valea Verde",
    "Codrii Noi",
    "Plai Senin",
    "Izvor Lin",
    "Dealul Mic",
    "Stejarul Batran",
    "Poiana Larga",
    "Fintina Rece",
    "Livada Mare",
    "Colina Alba",
    "Rasarit Nou",
    "Vatra Calda",
    "Podgoria Sud",
    "Busuioc",
    "Magura Inalta",
    "Lunca Lata",
    "Albina Harnica",
    "Cimpia Noua",
    "Piatra Sura",
    "Vinul Vechi",
    "Pescarus",
    "Liman Albastru",
    "Cetatea Veche",
    "Rauri Line",
    "Nucul Batran",
    "Teiul Inflorit",
    "Salcia Verde",
    "Ciresica",
    "Floare de Colt",
    "Mesteacan",
)

BUSINESS_KINDS = (
    ("Utilities", "utility"),
    ("Dental Clinic", "health"),
    ("Grocers", "retail"),
    ("Property Management", "housing"),
    ("Telecom", "utility"),
    ("Insurance", "finance"),
    ("Fitness Club", "leisure"),
    ("Bakery", "retail"),
    ("Auto Repair", "services"),
    ("Pharmacy", "health"),
    ("Books", "retail"),
    ("Energy", "utility"),
    ("Language School", "education"),
    ("Veterinary Practice", "health"),
    ("Hardware", "retail"),
    ("Travel", "leisure"),
    ("Furniture", "retail"),
    ("Logistics", "services"),
    ("Accounting", "services"),
    ("Cafe", "retail"),
)

# Generic, descriptive public-office names (P2G payments); they name no real institution.
PUBLIC_OFFICES = (
    "Local Tax Office (simulated)",
    "Vehicle Registration Office (simulated)",
    "Civil Status Office (simulated)",
    "Municipal Parking Service (simulated)",
    "Passport Services Office (simulated)",
    "Land Registry Office (simulated)",
    "Court Fees Office (simulated)",
    "Social Insurance Office (simulated)",
)

# Categories whose merchants accept MIA QR payments in the generator.
QR_CATEGORIES = frozenset({"retail", "leisure", "health", "services"})
BILLER_CATEGORIES = frozenset({"utility", "housing", "finance"})


@dataclass(frozen=True)
class AccountIdentity:
    account_id: str
    display_name: str
    account_type: str  # "personal" | "business"
    home_country: str
    alias: str | None  # MIA phone alias of a personal account; businesses are paid by IBAN or QR


@dataclass(frozen=True)
class CounterpartyIdentity:
    counterparty_id: str
    display_name: str
    kind: str  # "person" | "business" | "public_institution"
    category: str
    alias: str | None = None  # MIA phone alias of a person counterparty
    home_country: str = HOME_COUNTRY


class AliasBook:
    """Unique MIA-style phone aliases (`+373` + 8 digits) drawn from a seeded generator."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.used: set[str] = set()

    def new(self) -> str:
        while True:
            prefix = ("6", "7")[int(self.rng.integers(2))]
            alias = f"+373{prefix}{int(self.rng.integers(0, 10_000_000)):07d}"
            if alias not in self.used:
                self.used.add(alias)
                return alias


def _unique_person_names(rng: np.random.Generator, count: int, exclude: set[str]) -> list[str]:
    names: list[str] = []
    seen = set(exclude)
    while len(names) < count:
        name = f"{FIRST_NAMES[rng.integers(len(FIRST_NAMES))]} {LAST_NAMES[rng.integers(len(LAST_NAMES))]}"
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def build_accounts(seed: int = IDENTITY_SEED, count: int = ACCOUNT_COUNT) -> list[AccountIdentity]:
    """Return `count` SecFlow accounts with unique, stable display names and personal aliases."""
    rng = np.random.default_rng(seed)
    aliases = AliasBook(np.random.default_rng(seed + 7))
    person_names = _unique_person_names(rng, count, exclude=set())
    accounts: list[AccountIdentity] = []
    business_count = 0
    for index in range(count):
        is_business = rng.random() < 0.14
        if is_business:
            prefix = BUSINESS_PREFIXES[business_count % len(BUSINESS_PREFIXES)]
            label, _ = BUSINESS_KINDS[(business_count * 7 + 3) % len(BUSINESS_KINDS)]
            surname = person_names[index].split(" ", 1)[1]
            name = f"{prefix} {label}" if business_count < len(BUSINESS_PREFIXES) else f"{surname} {label}"
            suffix = 2
            base_name = name
            while name in {a.display_name for a in accounts} or name in person_names:
                name = f"{base_name} {suffix}"
                suffix += 1
            business_count += 1
        else:
            name = person_names[index]
        accounts.append(
            AccountIdentity(
                account_id=f"ACC-{index + 1:04d}",
                display_name=name,
                account_type="business" if is_business else "personal",
                home_country=HOME_COUNTRY,
                alias=None if is_business else aliases.new(),
            )
        )
    names = [a.display_name for a in accounts]
    if len(set(names)) != len(names):  # pragma: no cover - deterministic guard
        raise RuntimeError("Generated account names are not unique")
    return accounts


def build_counterparties(
    rng: np.random.Generator,
    count: int,
    id_prefix: str,
    start_index: int,
    reserved_names: set[str],
    aliases: AliasBook,
    business_share: float = 0.55,
    width: int = 5,
) -> list[CounterpartyIdentity]:
    """Generate external MIA participants with understandable display names."""
    result: list[CounterpartyIdentity] = []
    used = set(reserved_names)
    index = start_index
    while len(result) < count:
        if rng.random() < business_share:
            prefix = BUSINESS_PREFIXES[rng.integers(len(BUSINESS_PREFIXES))]
            label, category = BUSINESS_KINDS[rng.integers(len(BUSINESS_KINDS))]
            name, kind, alias = f"{prefix} {label}", "business", None
        else:
            name = (
                f"{FIRST_NAMES[rng.integers(len(FIRST_NAMES))]} {LAST_NAMES[rng.integers(len(LAST_NAMES))]}"
            )
            kind, category, alias = "person", "individual", None
        if name in used:
            # Names repeat in real life; a numeric suffix keeps display names unique here.
            name = f"{name} {index % 97 + 2}"
            if name in used:
                continue
        used.add(name)
        if kind == "person":
            alias = aliases.new()
        result.append(CounterpartyIdentity(f"{id_prefix}{index:0{width}d}", name, kind, category, alias))
        index += 1
    return result
