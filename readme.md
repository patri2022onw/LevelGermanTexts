# 🇩🇪 German Language Analyzer

A comprehensive web application for analyzing German texts based on CEFR levels (A1-C1), designed for language instructors and learners. Powered by **simplemma** for fast, dependency-free German lemmatization and **NLTK** for German-enhanced named entity recognition.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![Simplemma](https://img.shields.io/badge/simplemma-dependency--free-green.svg)
![NLTK](https://img.shields.io/badge/nltk-NER-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

- **CEFR Level Analysis**: Automatically identify words above selected proficiency levels
- **Fast Lemmatization**: Uses simplemma for dependency-free German word processing
- **Advanced Named Entity Recognition**: Uses NLTK with German-specific enhancements for detection of persons, locations, and organizations
- **Two Analysis Modes**:
  - **Leveling**: Simplify texts by removing/replacing difficult words
  - **Labeling**: Generate vocabulary lists with translations
- **AI-Powered Features**: 
  - Text simplification using Claude or Gemini
  - High-quality translations in 6 languages
  - Context-aware word processing
- **Multi-language Translation**: AI-powered translations to English, French, Spanish, Italian, Polish, and Russian
- **Intelligent Filtering**: Automatically exclude named entities and core German words
- **Batch Processing**: CLI tool for processing multiple files
- **Graceful Degradation**: Falls back to enhanced heuristic NER if NLTK unavailable

## 🚀 Quick Start

### Option 1: Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/german-language-analyzer.git
   cd german-language-analyzer
   ```

2. **Run the setup script**
   
   **For Linux/Mac:**
   ```bash
   chmod +x quickstart.sh
   ./quickstart.sh
   ```
   
   **For Windows:**
   ```batch
   quickstart.bat
   ```

3. **Add vocabulary files**
   - Place your vocabulary CSV files in the `vocabulary/` directory
   - Required files: `A1.csv`, `A2.csv`, `B1.csv`, `B2.csv`, `C1_withduplicates.csv`
   - Also add `german_stopwords_plain.txt` to the project root

4. **Run the application**
   ```bash
   # For original Flair-based NER
   streamlit run app.py
   
   # For NLTK-based NER
   streamlit run streamlit_app.py
   ```

### Option 2: Deploy on Streamlit Cloud

1. Fork this repository
2. Add all vocabulary files to your fork
3. Go to [share.streamlit.io](https://share.streamlit.io)
4. Connect your GitHub and deploy

## 📁 Project Structure

```
german-language-analyzer/
│
├── app.py                      # Main Streamlit application (Flair-based NER)
├── streamlit_app.py           # NLTK-based NER version
├── requirements.txt            # Python dependencies
├── translation_service.py      # Enhanced translation service
├── cli_batch_processor.py      # CLI for batch processing
├── check_files.py             # File verification script
│
├── quickstart.sh              # Setup script (Linux/Mac) - downloads NLTK data
├── quickstart.bat             # Setup script (Windows) - downloads NLTK data
│
├── .streamlit/
│   ├── config.toml           # Streamlit configuration
│   └── secrets.toml          # API keys (create locally)
│
├── german_stopwords_plain.txt # German stopwords
├── vocabulary/                # Vocabulary files directory
│   ├── A1.csv                # A1 vocabulary
│   ├── A2.csv                # A2 vocabulary
│   ├── B1.csv                # B1 vocabulary
│   ├── B2.csv                # B2 vocabulary
│   └── C1_withduplicates.csv # C1 vocabulary
```

## 🔧 Configuration

### API Keys

The app uses AI services for both text simplification and translation. Add your API keys to `.streamlit/secrets.toml`:

```toml
[api_keys]
claude_api_key = "your-claude-api-key"
gemini_api_key = "your-gemini-api-key"
```

### Vocabulary File Format

CSV files should have at least a `Lemma` column:
```csv
Lemma,Wortart,Genus,Artikel
Haus,Substantiv,neut.,das
arbeiten,Verb,,
```

## 📊 Usage

### Web Interface

1. **Upload/Auto-load Files**: The app automatically detects existing vocabulary files
2. **Input Text**: Type, paste, or upload German text
3. **Configure Analysis**:
   - Select target CEFR level
   - Choose Leveling or Labeling mode
   - Select translation language (for Labeling)
4. **Analyze**: Click the analyze button
5. **Export**: Download results as TXT or CSV

### Command Line (Batch Processing)

Process single file:
```bash
python cli_batch_processor.py -i text.txt -o output/ -l B1
```

Process directory:
```bash
python cli_batch_processor.py -i texts/ -o output/ -l A2 --batch
```

With AI-powered translation:
```bash
python cli_batch_processor.py -i text.txt -o output/ -l B2 -m labeling -t French --ai Claude
```

Using Gemini for translations:
```bash
python cli_batch_processor.py -i texts/ -o output/ -l B1 -m labeling -t Spanish --ai Gemini --batch
```

## 🌐 Translation Features

The app uses the same AI models (Claude or Gemini) for translations, providing:

- **High-quality translations** powered by advanced AI
- **Context-aware processing** for accurate word meanings
- **Batch translation** for efficient processing
- **Translation caching** to optimize API usage
- **Support for 6 languages**: English, French, Spanish, Italian, Polish, Russian

No additional translation API keys required - translations are included with the AI service!

## 🤖 AI Integration

### Claude
- Get API key from [Anthropic Console](https://console.anthropic.com/)
- Provides context-aware text simplification

### Gemini
- Get API key from [Google AI Studio](https://makersuite.google.com/)
- Alternative AI model for text processing

## 📈 Performance Tips

- **Fast Startup**: Simplemma works instantly, NLTK data downloaded once
- **Large Files**: Split texts over 10,000 words
- **Caching**: The app caches vocabulary mappings, translations, and NLTK data
- **Memory**: Simplemma uses minimal memory; NLTK data loaded once and cached
- **Batch Processing**: Use CLI for multiple files
- **NER Options**: Use fallback NER for faster processing if high accuracy not needed

## 🛠️ Development

### Running Tests
```bash
python -m pytest tests/
```

### Adding New Features
1. Create feature branch
2. Implement changes
3. Update documentation
4. Submit pull request

### Custom Word Lists
Create custom vocabulary lists following the CSV format with a `Lemma` column.

## 🐛 Troubleshooting

### Common Issues

**Import errors**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Simplemma works out-of-the-box, NLTK optional

**NLTK NER issues**
- If NLTK not installed: `pip install nltk`
- If NLTK data download fails, fallback NER will be used automatically
- First run may take longer due to NLTK data download

**File encoding issues**
- Ensure all files use UTF-8 encoding
- Check file line endings (LF vs CRLF)

**Memory errors**
- Reduce file size
- Increase Streamlit limits
- Use batch processing
- Consider using fallback NER for lower memory usage
- Use `streamlit_app.py` for NLTK-based NER (lower memory than Flair)

**Lemmatization issues**
- Simplemma works out-of-the-box for German
- No model training or configuration needed
- Check the word lookup tool in the sidebar for testing

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## 📚 Resources

- [CEFR Levels](https://www.coe.int/en/web/common-european-framework-reference-languages)
- [Simplemma Documentation](https://github.com/adbar/simplemma)
- [NLTK Documentation](https://www.nltk.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Anthropic Claude API](https://docs.anthropic.com/)
- [Google Gemini API](https://ai.google.dev/)

## 📞 Support

- Create an issue for bugs
- Start a discussion for features
- Check existing issues before posting

---

Made with ❤️ for German language educators and learners