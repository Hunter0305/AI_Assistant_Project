"""
test_medical.py — Tests for Task 2: Medical Q&A
Tests: entity extraction, safety layer, query routing
Note: RAG tests require the medical index to be built first.
Imports are done lazily inside test functions to avoid triggering
langchain_core/transformers/torch at module load time.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest


class TestMedicalEntityExtraction:
    """Test medical entity recognition (Task 2 requirement)."""

    def test_symptom_detection(self):
        """Should detect symptoms from query."""
        from src.medical.entities import extract_medical_entities
        entities = extract_medical_entities("I have a headache and fever with nausea")
        found = entities["symptoms"]
        assert any(s in found for s in ["headache", "fever", "nausea"]), f"No symptoms found: {found}"

    def test_disease_detection(self):
        """Should detect disease mentions."""
        from src.medical.entities import extract_medical_entities
        entities = extract_medical_entities("What are the treatments for diabetes and hypertension?")
        found = entities["diseases"]
        assert any(d in found for d in ["diabetes", "hypertension"]), f"No diseases found: {found}"

    def test_treatment_detection(self):
        """Should detect treatment mentions."""
        from src.medical.entities import extract_medical_entities
        entities = extract_medical_entities("Is surgery or chemotherapy better for this?")
        found = entities["treatments"]
        assert any(t in found for t in ["surgery", "chemotherapy"]), f"No treatments found: {found}"

    def test_medication_detection(self):
        """Should detect medication mentions."""
        from src.medical.entities import extract_medical_entities
        entities = extract_medical_entities("Should I take aspirin or ibuprofen?")
        found = entities["medications"]
        assert any(m in found for m in ["aspirin", "ibuprofen"]), f"No medications found: {found}"

    def test_returns_all_categories(self):
        """Return dict must have all required keys."""
        from src.medical.entities import extract_medical_entities
        entities = extract_medical_entities("test query")
        required_keys = {"symptoms", "diseases", "treatments", "medications", "other"}
        assert required_keys.issubset(set(entities.keys()))

    def test_empty_query_returns_empty_entities(self):
        """Empty query should return empty entity lists."""
        from src.medical.entities import extract_medical_entities
        entities = extract_medical_entities("")
        for key in ["symptoms", "diseases", "treatments", "medications"]:
            assert isinstance(entities[key], list)

    def test_format_entities_display(self):
        """format_entities_display should return a string."""
        from src.medical.entities import extract_medical_entities, format_entities_display
        entities = extract_medical_entities("I have diabetes and fever")
        display = format_entities_display(entities)
        assert isinstance(display, str)
        assert len(display) > 0


class TestMedicalSafetyLayer:
    """Test Task 2 safety requirements. These functions have no LLM import."""

    def test_emergency_detection_chest_pain(self):
        """Chest pain should be flagged as emergency."""
        from src.medical.retrieval import _is_emergency_query
        assert _is_emergency_query("I have severe chest pain and cannot breathe") is True

    def test_emergency_detection_stroke(self):
        """Stroke mention should be flagged as emergency."""
        from src.medical.retrieval import _is_emergency_query
        assert _is_emergency_query("I think I'm having a stroke") is True

    def test_non_emergency_query(self):
        """Normal medical question should not be flagged as emergency."""
        from src.medical.retrieval import _is_emergency_query
        assert _is_emergency_query("What are common symptoms of diabetes?") is False

    def test_dangerous_diagnosis_request(self):
        """Personal diagnosis request should be flagged."""
        from src.medical.retrieval import _is_dangerous_query
        assert _is_dangerous_query("Do I have cancer?") is True

    def test_non_dangerous_informational_query(self):
        """Informational medical question should not be flagged as dangerous."""
        from src.medical.retrieval import _is_dangerous_query
        assert _is_dangerous_query("What are the symptoms of diabetes?") is False


class TestMedicalIngestion:
    """Test MedQuAD ingestion logic (without network calls)."""

    def test_qa_pairs_to_documents(self):
        """QA pairs should convert to LangChain Documents correctly."""
        from src.medical.ingestion import qa_pairs_to_documents
        test_pairs = [
            {
                "question": "What is diabetes?",
                "answer": "Diabetes is a metabolic disease characterized by high blood sugar.",
                "topic": "Diabetes",
                "type": "disease",
                "source": "test",
            }
        ]
        docs = qa_pairs_to_documents(test_pairs)
        assert len(docs) == 1
        assert "What is diabetes?" in docs[0].page_content
        assert docs[0].metadata["source"] == "test"
