"""The five shared PHI labels and how each source's categories map onto them."""

# shared label -> ASQ-PHI identifier types
SHARED_TAXONOMY: dict[str, list[str]] = {
    "NAME":     ["NAME"],
    "DATE":     ["DATE"],
    "LOCATION": ["GEOGRAPHIC_LOCATION"],
    "CONTACT":  ["PHONE_NUMBER", "FAX_NUMBER", "EMAIL_ADDRESS", "IP_ADDRESS"],
    "ID":       ["MEDICAL_RECORD_NUMBER", "HEALTH_PLAN_BENEFICIARY_NUMBER", "SOCIAL_SECURITY_NUMBER",
                 "UNIQUE_IDENTIFIER", "ACCOUNT_NUMBER", "CERTIFICATE_LICENSE_NUMBER"],
}

SHARED_LABELS: list[str] = list(SHARED_TAXONOMY)

ASQ_TO_SHARED: dict[str, str] = {
    asq_type: label for label, asq_types in SHARED_TAXONOMY.items() for asq_type in asq_types
}

# Presidio entity types we keep; all others (e.g. URL, NRP) are dropped.
PRESIDIO_TO_SHARED: dict[str, str] = {
    "PERSON": "NAME",
    "DATE_TIME": "DATE",
    "LOCATION": "LOCATION",
    "PHONE_NUMBER": "CONTACT", "EMAIL_ADDRESS": "CONTACT", "IP_ADDRESS": "CONTACT",
    "US_SSN": "ID", "MEDICAL_LICENSE": "ID", "US_DRIVER_LICENSE": "ID", "US_PASSPORT": "ID",
    "US_BANK_NUMBER": "ID", "US_ITIN": "ID", "CREDIT_CARD": "ID", "IBAN_CODE": "ID", "UK_NHS": "ID",
}
