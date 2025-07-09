#!/bin/bash
# quickstart.sh - Quick setup script for German Language Analyzer

echo "🇩🇪 German Language Analyzer - Quick Start Setup"
echo "=============================================="

# Check Python version
echo -e "\n📍 Checking Python version..."
python_version=$(python3 --version 2>&1)
if [[ $? -eq 0 ]]; then
    echo "✅ Found: $python_version"
else
    echo "❌ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment
echo -e "\n📍 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "ℹ️  Virtual environment already exists"
fi

# Activate virtual environment
echo -e "\n📍 Activating virtual environment..."
source venv/bin/activate || . venv/Scripts/activate
echo "✅ Virtual environment activated"

# Install requirements
echo -e "\n📍 Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Requirements installed"

# Check for NLTK installation and download NER data
echo -e "\n📍 Setting up NLTK NER data..."
python -c "
try:
    import nltk
    print('ℹ️  NLTK detected, downloading required data packages...')
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
    print('✅ NLTK data packages downloaded and cached')
except ImportError:
    print('⚠️  NLTK not installed - using fallback NER')
    print('   Install NLTK for better NER: pip install nltk')
except Exception as e:
    print(f'⚠️  Could not download NLTK data: {e}')
    print('   Fallback NER will be used')
"

# Note: Simplemma is dependency-free
echo -e "\n📍 Simplemma setup..."
echo "✅ Simplemma is ready to use (no model download required)"

# Create directories if needed
echo -e "\n📍 Setting up project structure..."
mkdir -p .streamlit
mkdir -p logs
mkdir -p output
echo "✅ Directories created"

# Check for vocabulary files
echo -e "\n📍 Checking vocabulary files..."
python check_files.py

# Create example secrets file if it doesn't exist
if [ ! -f ".streamlit/secrets.toml" ]; then
    echo -e "\n📍 Creating example secrets file..."
    cat > .streamlit/secrets.toml << 'EOF'
# Example secrets file - Add your actual API keys here
[api_keys]
# claude_api_key = "your-claude-api-key"
# gemini_api_key = "your-gemini-api-key"
# deepl_api_key = "your-deepl-api-key"

[deployment]
environment = "development"
EOF
    echo "✅ Created .streamlit/secrets.toml (add your API keys)"
fi

# Final instructions
echo -e "\n✨ Setup complete!"
echo -e "\n📋 Next steps:"
echo "1. Add your vocabulary files (A1.csv, A2.csv, etc.) to the vocabulary/ directory"
echo "2. Add your API keys to .streamlit/secrets.toml (optional)"
echo "3. Run the app with: streamlit run app.py (Flair NER) or streamlit run streamlit_app.py (NLTK NER)"
echo -e "\n💡 Features:"
echo "   - Uses simplemma for fast, dependency-free German lemmatization"
echo "   - Uses NLTK with German enhancements for named entity recognition"
echo "   - Falls back to enhanced heuristic NER if NLTK not available"
echo "   - Optimized for Python 3.13 compatibility"
echo "   - Supports Claude and Gemini AI integration"
echo -e "\n💡 For deployment on Streamlit Cloud:"
echo "   - Push all files to GitHub"
echo "   - Deploy at share.streamlit.io"
echo "   - Configure secrets in the Streamlit Cloud dashboard"