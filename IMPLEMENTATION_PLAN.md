# Implementation Plan - Tech Stack Updates

**Created:** January 2026
**Scope:** Update dependencies and modernize AI service integrations

---

## Overview

This plan addresses updates identified in `TECH_STACK_REVIEW.md` across 3 phases:
- **Phase 1:** Critical SDK and model updates (HIGH priority)
- **Phase 2:** Dependency version range updates (MEDIUM priority)
- **Phase 3:** Optional improvements (LOW priority)

---

## Phase 1: Critical Updates

### Priority: HIGH | Risk: MEDIUM

#### Step 1.1: Update AI SDK Versions in requirements.txt

**File:** `requirements.txt` (lines 19-21)

| Package | Current | Target |
|---------|---------|--------|
| anthropic | `>=0.7.0` | `>=0.40.0,<1.0.0` |
| google-genai | `>=0.7.0` | `>=1.0.0,<2.0.0` |

**Verification:**
```bash
pip install -r requirements.txt
python -c "import anthropic; from google import genai; print('OK')"
```

---

#### Step 1.2: Update Claude Model Identifiers

**Find and Replace:**
| Old Model | New Model |
|-----------|-----------|
| `claude-3-sonnet-20240229` | `claude-sonnet-4-20250514` |
| `claude-3-5-sonnet-20240620` | `claude-sonnet-4-20250514` |

**Files and Locations:**

| File | Lines | Function |
|------|-------|----------|
| `streamlit_app.py` | 426, 507, 582 | simplify_text_claude, _translate_with_claude, batch translate |
| `app.py` | 376, 457, 532 | simplify_text_claude, _translate_with_claude, batch translate |
| `translation_service.py` | 35 | ClaudeTranslationProvider.__init__ |

**Verification:**
1. Start app: `streamlit run streamlit_app.py`
2. Configure Claude API key
3. Test text simplification and translation

---

#### Step 1.3: Verify Google GenAI Compatibility

The codebase uses `genai.Client()` and `client.models.generate_content()` patterns. Verify these work with the updated SDK.

**Potential issue:** Response structure may have changed. Check if `response.text` still works or needs `response.candidates[0].content.parts[0].text`.

---

## Phase 2: Recommended Updates

### Priority: MEDIUM | Risk: LOW

#### Step 2.1: Update Dependency Ranges

**File:** `requirements.txt`

| Package | Current | Target | Reason |
|---------|---------|--------|--------|
| streamlit | `>=1.28.0,<2.0.0` | `>=1.35.0,<2.0.0` | Performance improvements |
| pandas | `>=2.0.0,<2.3.0` | `>=2.0.0,<3.0.0` | Relaxed upper bound |
| numpy | `>=1.26.0,<2.0.0` | `>=1.26.0,<3.0.0` | NumPy 2.x compatibility |
| flair | `>=0.13.0` | `>=0.14.0,<0.16.0` | Bug fixes |

**Verification:**
```bash
# Create fresh environment
python -m venv test_env && source test_env/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## Phase 3: Optional Improvements

### Priority: LOW | Risk: LOW

#### Step 3.1: Add Development Dependencies

Create `requirements-dev.txt`:
```txt
-r requirements.txt

# Type checking
mypy>=1.8.0
types-requests>=2.31.0

# Testing
pytest>=8.0.0
pytest-cov>=4.1.0
```

---

#### Step 3.2: Add Testing Framework

**Directory Structure:**
```
tests/
├── __init__.py
├── conftest.py
├── test_analyzer.py
├── test_translation.py
└── test_ner.py
```

---

## Updated requirements.txt (Complete)

```txt
# requirements.txt
# German Language Analyzer dependencies - Optimized for Python 3.13

# Core dependencies
streamlit>=1.35.0,<2.0.0
pandas>=2.0.0,<3.0.0
numpy>=1.26.0,<3.0.0

# Lemmatization
simplemma>=1.0.0

# NER for German - NLTK
nltk>=3.8.0

# Optional: Flair for app.py compatibility
flair>=0.14.0,<0.16.0
torch>=2.0.0

# AI services
anthropic>=0.40.0,<1.0.0
google-genai>=1.0.0,<2.0.0
requests>=2.31.0

# Core scientific packages
scipy>=1.11.0

# Build tools
setuptools>=65.0.0
wheel>=0.38.0
```

---

## Implementation Order

```
Phase 1 (Critical)
│
├─► Step 1.1: Update SDK versions in requirements.txt
│       │
│       ├─► Step 1.2: Update Claude model identifiers
│       │
│       └─► Step 1.3: Verify Google GenAI compatibility
│
▼
Phase 2 (Recommended) — After Phase 1 verified
│
├─► Step 2.1: Update core dependency ranges
│
└─► Step 2.2: Test NumPy 2.x compatibility (optional)
│
▼
Phase 3 (Optional) — Independent of Phase 2
│
├─► Step 3.1: Add type checking support
│
└─► Step 3.2: Add testing framework
```

---

## Verification Checklist

### Phase 1 Complete When:
- [ ] All dependencies install without errors
- [ ] Streamlit app starts without import errors
- [ ] Claude API calls succeed with new model identifier
- [ ] Gemini API calls succeed with updated SDK
- [ ] Text simplification works (both providers)
- [ ] Translation works (both providers)
- [ ] CLI batch processor runs without errors

### Phase 2 Complete When:
- [ ] Fresh install succeeds in new virtual environment
- [ ] No deprecation warnings from Streamlit
- [ ] Pandas/NumPy operations work correctly
- [ ] Flair NER model loads (app.py)
- [ ] NLTK NER works (streamlit_app.py)

### Phase 3 Complete When:
- [ ] mypy runs without critical errors
- [ ] pytest discovers and runs tests
- [ ] Code coverage report generates

---

## Risk Mitigation

| Change | Risk | Mitigation |
|--------|------|------------|
| Anthropic SDK | Medium | Pin to specific version if issues |
| Claude model | Low | Fallback to `claude-3-5-sonnet-latest` |
| Google GenAI | Medium | Check response structure |
| Streamlit | Low | Stay on 1.35.x |
| NumPy/Pandas | Low | Pin to 1.x if needed |

---

## Rollback Procedure

**Quick Rollback:**
```bash
git checkout HEAD~1 -- requirements.txt streamlit_app.py app.py translation_service.py
pip install -r requirements.txt --force-reinstall
```

**Model Rollback:**
Replace `claude-sonnet-4-20250514` with `claude-3-5-sonnet-20241022`

---

## Files Modified Summary

| File | Phase | Changes |
|------|-------|---------|
| `requirements.txt` | 1, 2 | Version constraints |
| `streamlit_app.py` | 1 | 3 model identifiers (lines 426, 507, 582) |
| `app.py` | 1 | 3 model identifiers (lines 376, 457, 532) |
| `translation_service.py` | 1 | 1 model identifier (line 35) |
| `requirements-dev.txt` | 3 | New file (optional) |
| `tests/*` | 3 | New directory (optional) |
