#!/bin/bash
# Streamlit deployment script

echo "🚀 RAG Chatbot Deployment Checker"
echo "=================================="

# Check Python version
echo "✅ Python Version:"
python --version

# Check if virtual environment is active
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual Environment: Active ($VIRTUAL_ENV)"
else
    echo "❌ Virtual Environment: Not active"
    echo "   Please activate: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)"
fi

# Check requirements
echo ""
echo "✅ Checking Requirements:"
if [ -f "requirements.txt" ]; then
    echo "   📄 requirements.txt found"
    echo "   📦 Installing dependencies..."
    pip install -r requirements.txt
else
    echo "   ❌ requirements.txt not found"
fi

# Check environment file
echo ""
echo "✅ Environment Configuration:"
if [ -f ".env.example" ]; then
    echo "   📄 .env.example found"
    if [ -f ".env" ]; then
        echo "   📄 .env found"
    else
        echo "   ⚠️  .env not found - copying from .env.example"
        cp .env.example .env
        echo "   📝 Please edit .env file and add your GEMINI_API_KEY"
    fi
else
    echo "   ❌ .env.example not found"
fi

# Check project structure
echo ""
echo "✅ Project Structure:"
required_files=("app.py" "src/rag_pipeline.py" "src/document_processor.py" "src/utils.py")
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✅ $file"
    else
        echo "   ❌ $file missing"
    fi
done

# Check directories
required_dirs=("src" "notebooks" "tests" "data")
for dir in "${required_dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "   ✅ $dir/"
    else
        echo "   ❌ $dir/ missing"
        mkdir -p "$dir"
        echo "   📁 Created $dir/"
    fi
done

echo ""
echo "🎯 Ready to Deploy!"
echo "Next steps:"
echo "1. Make sure .env file has your GEMINI_API_KEY"
echo "2. Test locally: streamlit run app.py"
echo "3. Push to GitHub: git push origin main"
echo "4. Deploy on Streamlit Cloud: https://share.streamlit.io/"
echo ""
echo "📖 GitHub Repository: https://github.com/selahmet/rag-chatbot"