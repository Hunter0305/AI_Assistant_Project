"""
entities.py — Medical entity recognition for Task 2.
Uses spaCy's en_core_web_sm model combined with curated medical term lists
to identify symptoms, diseases, treatments, medications, and conditions.

Design: spaCy is used for base NER (ORG, PERSON, etc.) and POS tagging.
Custom pattern matching identifies domain-specific medical entities.
No GPU required; en_core_web_sm is ~12MB.
"""
import re
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# ── Curated medical keyword sets ─────────────────────────────────────────────
SYMPTOM_KEYWORDS = {
    "pain", "ache", "fever", "fatigue", "nausea", "vomiting", "dizziness",
    "headache", "cough", "breathlessness", "swelling", "rash", "itching",
    "bleeding", "weakness", "numbness", "tingling", "chest pain", "back pain",
    "abdominal pain", "shortness of breath", "loss of appetite", "weight loss",
    "weight gain", "insomnia", "depression", "anxiety", "palpitations",
    "diarrhea", "constipation", "bloating", "jaundice", "dehydration",
}

DISEASE_KEYWORDS = {
    "diabetes", "cancer", "hypertension", "asthma", "arthritis", "alzheimer",
    "parkinson", "epilepsy", "stroke", "heart disease", "pneumonia", "tuberculosis",
    "hiv", "aids", "hepatitis", "malaria", "covid", "flu", "influenza",
    "kidney disease", "liver disease", "anemia", "thyroid", "lupus", "psoriasis",
    "multiple sclerosis", "celiac", "crohn", "ibs", "copd", "migraine",
    "fibromyalgia", "osteoporosis", "leukemia", "lymphoma", "melanoma",
}

TREATMENT_KEYWORDS = {
    "surgery", "therapy", "chemotherapy", "radiation", "immunotherapy",
    "dialysis", "transplant", "vaccine", "vaccination", "physiotherapy",
    "psychotherapy", "counseling", "rehabilitation", "exercise", "diet",
    "lifestyle", "medication", "treatment", "procedure", "intervention",
    "operation", "biopsy", "screening", "monitoring", "management",
}

MEDICATION_KEYWORDS = {
    "aspirin", "ibuprofen", "paracetamol", "acetaminophen", "metformin",
    "insulin", "antibiotic", "antibiotic", "steroid", "prednisone",
    "antidepressant", "antiviral", "antifungal", "antihistamine",
    "beta blocker", "ace inhibitor", "statin", "diuretic", "vaccine",
    "morphine", "opioid", "analgesic", "antacid", "laxative",
}


def _regex_entity_extract(text: str, keyword_set: set, label: str) -> List[Dict]:
    """Find keyword matches in text (case-insensitive)."""
    found = []
    text_lower = text.lower()
    for kw in keyword_set:
        if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
            found.append({"text": kw, "label": label})
    return found


def _spacy_ner(text: str) -> List[Dict]:
    """Run spaCy NER. Returns empty list if spaCy/model unavailable."""
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model en_core_web_sm not found. Run: python -m spacy download en_core_web_sm")
            return []
        doc = nlp(text)
        entities = []
        for ent in doc.ents:
            if ent.label_ in {"DISEASE", "DRUG", "CHEMICAL", "ORG", "GPE"}:
                entities.append({"text": ent.text, "label": ent.label_})
        return entities
    except ImportError:
        logger.warning("spaCy not installed. Install with: pip install spacy")
        return []


def extract_medical_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract medical entities from text.

    Returns a dict with keys:
        symptoms, diseases, treatments, medications, other
    Each value is a list of identified entity strings (deduplicated).
    """
    result: Dict[str, List[str]] = {
        "symptoms": [],
        "diseases": [],
        "treatments": [],
        "medications": [],
        "other": [],
    }

    symptoms = _regex_entity_extract(text, SYMPTOM_KEYWORDS, "SYMPTOM")
    diseases = _regex_entity_extract(text, DISEASE_KEYWORDS, "DISEASE")
    treatments = _regex_entity_extract(text, TREATMENT_KEYWORDS, "TREATMENT")
    medications = _regex_entity_extract(text, MEDICATION_KEYWORDS, "MEDICATION")

    result["symptoms"] = list({e["text"] for e in symptoms})
    result["diseases"] = list({e["text"] for e in diseases})
    result["treatments"] = list({e["text"] for e in treatments})
    result["medications"] = list({e["text"] for e in medications})

    # Add spaCy entities to 'other'
    spacy_ents = _spacy_ner(text)
    known = set(result["symptoms"] + result["diseases"] + result["treatments"] + result["medications"])
    result["other"] = list({e["text"] for e in spacy_ents if e["text"].lower() not in known})

    return result


def format_entities_display(entities: Dict[str, List[str]]) -> str:
    """Format extracted entities for display in Streamlit."""
    lines = []
    label_map = {
        "symptoms": "🤒 Symptoms",
        "diseases": "🏥 Conditions/Diseases",
        "treatments": "💊 Treatments",
        "medications": "💉 Medications",
        "other": "📌 Other Entities",
    }
    for key, label in label_map.items():
        items = entities.get(key, [])
        if items:
            lines.append(f"**{label}**: {', '.join(items)}")
    return "\n\n".join(lines) if lines else "No specific medical entities detected."
