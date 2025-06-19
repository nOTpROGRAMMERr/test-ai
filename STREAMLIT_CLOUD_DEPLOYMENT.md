# Streamlit Cloud Deployment Guide

## 🚀 API Key Management for Streamlit Cloud

### **Current Setup (After Updates):**

✅ **All configuration now uses Streamlit secrets first, then environment variables as fallback**

### **How It Works:**

1. **Streamlit Cloud**: Reads from `secrets.toml` (uploaded to secret manager)
2. **Local Development**: Falls back to environment variables or `.env` file
3. **Other Platforms**: Uses environment variables

## 📋 Deployment Steps for Streamlit Cloud

### **1. Update Your secrets.toml File**

Make sure your `.streamlit/secrets.toml` has all required API keys:

```toml
# API Keys - Replace with your actual keys
OPENAI_API_KEY = "your_actual_openai_key"
COHERE_API_KEY = "your_actual_cohere_key"
GROQ_API_KEY = "your_actual_groq_key"
XAI_API_KEY = "your_actual_xai_key"
GOOGLE_API_KEY = "your_actual_google_key"
UPSTAGE_API_KEY = "your_actual_upstage_key"
PINECONE_API_KEY = "your_actual_pinecone_key"
LANGCHAIN_API_KEY = "your_actual_langchain_key"

# Database Configuration
PINECONE_INDEX_NAME = "profile-chunks"
PINECONE_ENVIRONMENT = "your_pinecone_environment_url"
AWS_ACCESS_KEY_ID = "your_aws_access_key"
AWS_SECRET_ACCESS_KEY = "your_aws_secret_key"
AWS_REGION = "ap-northeast-1"

# Application Configuration
LOG_LEVEL = "INFO"
EMBEDDING_MODEL = "openai"
RERANK_MODEL = "rerank-english-v3.0"
PROFILE_BONUS_ALPHA = 0.05
PROFILE_SCORE_THRESHOLD = 0.70
TOP_K_PROFILES = 20
MIN_PROFILE_SCORE = 0.80

# Authentication (already configured)
[auth]
cookie_name = "Test_recruitment_app"
cookie_key = "Test_secure_cookie_key_2024"
cookie_expiry_days = 7

[auth.users.admin]
name = "Administrator"
password = "LJIuBGiBKMBVLjSaC8Fa"
email = "admin@yourcompany.com"

[auth.users.mithun]
name = "Mithun"
password = "aVccZvFUYbNxPVl1nKYG"
email = "hr@yourcompany.com"

[auth.users.recruiter]
name = "Recruiter"
password = "FbTKCwq7K5UcOBxgsGo5"
email = "recruiter@yourcompany.com"

[auth.users.ld]
name = "Lakshman"
password = "2TkBnskJrgtMifeXv3BS"
email = "lakshman@yourcompany.com"
```

### **2. Deploy to Streamlit Cloud**

1. **Push your code** to GitHub (without the `secrets.toml` file - it's in `.gitignore`)
2. **Go to Streamlit Cloud** and create a new app
3. **Upload secrets.toml** to the secret manager in Streamlit Cloud
4. **Deploy** - all secrets will be automatically loaded

### **3. Verify Deployment**

The app will automatically:
- ✅ Load all API keys from Streamlit secrets
- ✅ Use authentication credentials from secrets
- ✅ Configure all application settings from secrets

## 🔐 Security Benefits

### **For Streamlit Cloud:**
- **✅ All secrets in secret manager** - No credentials in code or repository
- **✅ Encrypted storage** - Streamlit Cloud encrypts all secrets
- **✅ Access control** - Only authorized users can view/edit secrets
- **✅ Audit trail** - Track who accesses secrets

### **For Local Development:**
- **✅ Flexible setup** - Use either secrets.toml or .env file
- **✅ No conflicts** - Secrets take priority, env vars as fallback
- **✅ Easy testing** - Same code works in both environments

## 🛠️ Code Changes Made

### **Updated Files:**
1. **`utils.py`** - Now reads API keys from `st.secrets` first
2. **`post_rerank_aggregator.py`** - Configuration from secrets first
3. **`auth_config.py`** - Already using secrets for authentication

### **Smart Fallback System:**
```python
# Example pattern used throughout the codebase
try:
    api_key = st.secrets["API_KEY"]  # Streamlit Cloud
except:
    api_key = os.getenv("API_KEY")   # Local development
```

## 🚀 Ready for Production

Your app is now **fully optimized for Streamlit Cloud deployment**:

- **🔐 Secure** - All credentials in secret manager
- **🔄 Flexible** - Works locally and in cloud
- **📱 Scalable** - Ready for production use
- **🛡️ Compliant** - Follows security best practices

Just update your API keys in `secrets.toml` and deploy! 🎉 