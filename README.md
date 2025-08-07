# ScholarMate 📚🤖

> Transform any educational content into interactive learning tools powered by AI

ScholarMate is a full-stack AI application that converts educational content from multiple sources (PDFs, YouTube videos, or raw text) into comprehensive learning tools including summaries, glossaries, Q&A sets, and interactive MCQ quizzes.

## ✨ Features

### 🎯 Multi-Modal Content Input
- **PDF Processing**: Upload and extract text from PDF documents
- **YouTube Integration**: Fetch transcripts directly from YouTube videos
- **Direct Text Input**: Paste any text content for immediate processing

### 🧠 AI-Powered Learning Tools
- **Smart Summarization**: Generate topic-wise summaries using advanced map-reduce strategies
- **Technical Glossary**: Automatically identify and define key technical terms
- **Q&A Generation**: Create comprehensive question-and-answer sets for self-study
- **Interactive MCQ Quizzes**: Dynamic multiple-choice tests with instant scoring and feedback

### 🚀 Advanced Features
- **Intelligent Caching**: Prevents redundant API calls and speeds up performance
- **Content Change Detection**: Automatically detects when input changes and refreshes results
- **Interactive UI**: Clean, tab-based interface built with Streamlit
- **Real-time Scoring**: Instant quiz feedback with retake options

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   FastAPI        │    │   GROQ API      │
│   Frontend      │◄──►│   Backend        │◄──►│   (Gemma Model) │
│                 │    │                  │    │                 │
│ • File Upload   │    │ • Text Extraction│    │ • AI Processing │
│ • Tab Interface │    │ • API Endpoints  │    │ • LangChain     │
│ • Quiz System   │    │ • State Management│   │ • Content Gen   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Tech Stack
- **Frontend**: Streamlit
- **Backend**: FastAPI
- **AI Framework**: LangChain
- **AI Model**: Gemma (via GROQ API)
- **File Processing**: Custom PDF loader utilities
- **State Management**: Streamlit session state

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- GROQ API Key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/scholarmate.git
cd scholarmate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create a .env file
GROQ_API_KEY=your_groq_api_key_here
```

4. Start the backend server:
```bash
uvicorn main:app --reload
```

5. Launch the frontend:
```bash
streamlit run app.py
```

## 📖 Usage

### 1. Choose Your Content Source
- **Upload PDF**: Click "Browse files" to upload educational PDFs
- **YouTube URL**: Paste any YouTube video URL to extract transcripts
- **Direct Text**: Copy and paste text content directly

### 2. Generate Learning Materials
Navigate through the tabs to access different tools:
- **Summary**: Get topic-wise breakdowns of your content
- **Glossary**: View automatically generated technical definitions
- **Q&A**: Practice with generated question-and-answer pairs
- **MCQ Quiz**: Take interactive multiple-choice tests

### 3. Interactive Learning
- Take quizzes with shuffled options
- Get instant feedback and scoring
- Retake tests or generate new questions
- Switch between different learning modes seamlessly

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/extract_text/` | POST | Extract text from uploaded PDF files |
| `/get_youtube_transcript/` | POST | Fetch transcript from YouTube URL |
| `/summarize_document/` | POST | Generate topic-wise summaries |
| `/generate_glossary/` | POST | Create technical glossary |
| `/generate_qa/` | POST | Generate Q&A pairs |
| `/generate_mcq/` | POST | Create MCQ quiz questions |

## 🎯 Key Implementation Highlights

### Intelligent Caching System
```python
# Content hash-based caching prevents redundant API calls
if st.session_state.get('last_processed_content_hash') != content_hash:
    # Clear old cache and process new content
    clear_cache()
    process_new_content()
```

### Dynamic Quiz State Management
- Real-time answer tracking
- Automatic scoring calculation
- Session-based state persistence
- Interactive feedback system

### Robust Error Handling
- Graceful API failure recovery
- User-friendly error messages
- Temporary file cleanup
- Connection timeout management

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **GROQ** for providing the AI API infrastructure
- **Google** for the open-source Gemma model
- **LangChain** for AI orchestration capabilities
- **Streamlit** for the intuitive frontend framework
- **FastAPI** for the robust backend architecture

## 📊 Project Stats

- ✅ Multi-modal content processing
- ✅ Real-time AI-powered content generation
- ✅ Interactive quiz system with scoring
- ✅ Intelligent caching and state management
- ✅ Responsive web interface
- ✅ RESTful API architecture

## 🔮 Future Enhancements

- [ ] Support for additional file formats (DOCX, PPT)
- [ ] Advanced analytics and learning progress tracking
- [ ] Collaborative study sessions
- [ ] Mobile app development
- [ ] Integration with popular LMS platforms

---

**Made with ❤️ for learners everywhere**

*Transform your study materials into interactive learning experiences with ScholarMate!*
