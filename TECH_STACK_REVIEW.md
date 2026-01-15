# Tech Stack Review - German Language Analyzer

**Review Date:** January 2026
**Python Version:** 3.8+ (optimized for 3.13)
**Project Type:** Streamlit Web Application

---

## Executive Summary

The codebase uses a solid foundation of Python/Streamlit with NLP libraries for German text analysis. However, several dependencies require updates, particularly the AI service SDKs which are significantly outdated. This review identifies **3 critical**, **4 recommended**, and **2 optional** updates.

---

## Critical Updates Required

### 1. Anthropic SDK (HIGH PRIORITY)

| Current | Recommended | Latest |
|---------|-------------|--------|
| `>=0.7.0` | `>=0.40.0` | 0.44+ |

**Issue:** The Anthropic Python SDK has undergone major changes since version 0.7.0. The current constraint allows very old SDK versions that may have:
- Deprecated API patterns
- Missing features (streaming improvements, tool use, etc.)
- Potential security vulnerabilities

**Code Impact:**
- `streamlit_app.py:11` - `import anthropic`
- `app.py:11` - `import anthropic`
- `translation_service.py:30-57` - ClaudeTranslationProvider

**Recommendation:**
```diff
- anthropic>=0.7.0
+ anthropic>=0.40.0,<1.0.0
```

### 2. Claude Model Versions (HIGH PRIORITY)

The codebase uses outdated Claude model identifiers:

| Location | Current Model | Recommended Model |
|----------|---------------|-------------------|
| `streamlit_app.py:426` | `claude-3-5-sonnet-20240620` | `claude-sonnet-4-20250514` |
| `streamlit_app.py:507` | `claude-3-5-sonnet-20240620` | `claude-sonnet-4-20250514` |
| `streamlit_app.py:582` | `claude-3-sonnet-20240229` | `claude-sonnet-4-20250514` |
| `app.py:376` | `claude-3-5-sonnet-20240620` | `claude-sonnet-4-20250514` |
| `app.py:457` | `claude-3-5-sonnet-20240620` | `claude-sonnet-4-20250514` |
| `app.py:532` | `claude-3-sonnet-20240229` | `claude-sonnet-4-20250514` |
| `translation_service.py:35` | `claude-3-sonnet-20240229` | `claude-sonnet-4-20250514` |

**Notes:**
- Claude Sonnet 4 (`claude-sonnet-4-20250514`) is the current production model
- For cost-sensitive operations, consider `claude-haiku-4-20250514`
- The older models may be deprecated soon

### 3. Google GenAI SDK (HIGH PRIORITY)

| Current | Recommended | Latest |
|---------|-------------|--------|
| `>=0.7.0` | `>=1.0.0` | 1.x |

**Issue:** The Google GenAI SDK has undergone significant changes. The import style and API patterns may differ.

**Code Impact:**
- `streamlit_app.py:12` - `from google import genai`
- `app.py:12` - `from google import genai`

**Recommendation:**
```diff
- google-genai>=0.7.0
+ google-genai>=1.0.0,<2.0.0
```

---

## Recommended Updates

### 4. Streamlit Version Range (MEDIUM)

| Current | Recommended |
|---------|-------------|
| `>=1.28.0,<2.0.0` | `>=1.35.0,<2.0.0` |

**Rationale:** Streamlit 1.35+ includes performance improvements, better caching (`@st.cache_data`), and bug fixes. The upper bound of `<2.0.0` is appropriate for stability.

### 5. Pandas Version Range (MEDIUM)

| Current | Recommended |
|---------|-------------|
| `>=2.0.0,<2.3.0` | `>=2.0.0,<3.0.0` |

**Rationale:** The upper bound `<2.3.0` is unnecessarily restrictive. Pandas 2.x maintains backward compatibility within the major version.

### 6. NumPy Version Range (MEDIUM)

| Current | Recommended |
|---------|-------------|
| `>=1.26.0,<2.0.0` | `>=1.26.0,<3.0.0` |

**Rationale:** NumPy 2.0 was released in 2024 with some breaking changes, but most codebases are compatible. Consider testing with NumPy 2.x as dependencies (pandas, scipy) now support it.

### 7. Flair NER Library (MEDIUM)

| Current | Recommended |
|---------|-------------|
| `>=0.13.0` | `>=0.14.0,<0.16.0` |

**Rationale:** Flair 0.14+ includes bug fixes and improved German NER models. Note that Flair is optional in this project (NLTK provides fallback).

---

## Optional Updates

### 8. Add Type Checking Support (LOW)

Consider adding type checking dependencies for development:

```txt
# Development dependencies (optional)
mypy>=1.8.0
types-requests>=2.31.0
```

### 9. Add Testing Framework (LOW)

No testing framework is currently configured. Consider adding:

```txt
# Testing (optional)
pytest>=8.0.0
pytest-cov>=4.1.0
```

---

## Dependency Security Notes

### Current Secure Versions
- `requests>=2.31.0` - Good, includes security fixes
- `setuptools>=65.0.0` - Good, addresses known vulnerabilities
- `torch>=2.0.0` - Good, but heavy (~2GB)

### Packages to Monitor
- `nltk` - Ensure data downloads are from official sources
- `anthropic`, `google-genai` - Keep updated for API security

---

## Proposed Updated requirements.txt

```txt
# requirements.txt
# German Language Analyzer dependencies - Optimized for Python 3.13

# Core dependencies
streamlit>=1.35.0,<2.0.0
pandas>=2.0.0,<3.0.0
numpy>=1.26.0,<3.0.0

# Lemmatization - using simplemma (Python 3.13 compatible)
simplemma>=1.0.0

# NER for German - NLTK (Python 3.13 compatible)
nltk>=3.8.0

# Optional: Keep Flair for app.py compatibility
flair>=0.14.0,<0.16.0
torch>=2.0.0  # Required by Flair

# AI services (UPDATED)
anthropic>=0.40.0,<1.0.0
google-genai>=1.0.0,<2.0.0
requests>=2.31.0

# Core scientific packages
scipy>=1.11.0

# Build tools (important for Python 3.12+)
setuptools>=65.0.0
wheel>=0.38.0
```

---

## Model Version Update Locations

To update Claude model versions, modify these files:

1. **streamlit_app.py** - Lines 426, 507, 582
2. **app.py** - Lines 376, 457, 532
3. **translation_service.py** - Line 35

Suggested find/replace:
- `claude-3-sonnet-20240229` -> `claude-sonnet-4-20250514`
- `claude-3-5-sonnet-20240620` -> `claude-sonnet-4-20250514`

---

## Migration Path

### Phase 1: SDK Updates (Minimal Code Changes)
1. Update `requirements.txt` with new version constraints
2. Test AI service initialization
3. Update Claude model identifiers

### Phase 2: Code Improvements (Optional)
1. Add type hints to core modules
2. Add pytest configuration
3. Create CI/CD pipeline

---

## Compatibility Matrix

| Dependency | Python 3.8 | Python 3.11 | Python 3.13 |
|------------|------------|-------------|-------------|
| streamlit 1.35+ | Yes | Yes | Yes |
| pandas 2.x | Yes | Yes | Yes |
| numpy 1.26+ | Yes | Yes | Yes |
| anthropic 0.40+ | Yes | Yes | Yes |
| google-genai 1.x | Yes | Yes | Yes |
| flair 0.14+ | Yes | Yes | Yes |
| torch 2.x | Yes | Yes | Yes |

---

## Summary

| Priority | Update | Impact | Effort |
|----------|--------|--------|--------|
| Critical | Anthropic SDK 0.40+ | High | Low |
| Critical | Claude model versions | High | Low |
| Critical | Google GenAI SDK 1.x | High | Low |
| Recommended | Streamlit 1.35+ | Medium | Low |
| Recommended | Pandas version range | Low | Low |
| Recommended | NumPy version range | Low | Low |
| Recommended | Flair 0.14+ | Low | Low |
| Optional | Type checking | Low | Medium |
| Optional | Testing framework | Low | Medium |

**Total Estimated Files to Modify:** 4 (requirements.txt, streamlit_app.py, app.py, translation_service.py)
