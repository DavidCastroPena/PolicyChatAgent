# Deploying PolicyChatAgent to Streamlit Cloud

## Important Note About Qdrant Database

This application uses Qdrant vector database for local paper storage. **Streamlit Cloud deployment will work ONLY with the "External Papers Only" mode** unless you set up an external Qdrant instance.

## Deployment Options

### Option 1: Deploy with External Papers Only (Recommended for Demo)

This is the simplest option and requires no additional infrastructure.

#### Steps:

1. **Go to Streamlit Cloud**
   - Visit https://share.streamlit.io/
   - Sign in with your GitHub account

2. **Deploy New App**
   - Click "New app"
   - Repository: `DavidCastroPena/PolicyChatAgent`
   - Branch: `main`
   - Main file path: `ux.py`

3. **Configure Secrets**
   Click "Advanced settings" → "Secrets" and add:
   ```toml
   GEMINI_API_KEY = "your-gemini-api-key-here"
   SEMANTIC_SCHOLAR_API_KEY = "your-semantic-scholar-api-key-here"
   ```

4. **Deploy**
   - Click "Deploy"
   - Wait for deployment (2-3 minutes)
   - Your app will be live at `https://[your-app-name].streamlit.app`

5. **Usage**
   - Users must select "External papers only" option when using the app
   - Local papers options will not work without Qdrant

---

### Option 2: Deploy with Cloud Qdrant (Full Functionality)

For production use with local paper support, you need a cloud Qdrant instance.

#### Steps:

1. **Set up Qdrant Cloud**
   - Go to https://cloud.qdrant.io/
   - Create a free account
   - Create a new cluster
   - Note your cluster URL and API key

2. **Update Code for Cloud Qdrant**
   Modify `retriever/QdrantCollection.py` to accept URL and API key:
   ```python
   def __init__(self, host="localhost", port=6333, collection_name="papers", url=None, api_key=None):
       if url and api_key:
           self.client = QdrantClient(url=url, api_key=api_key)
       else:
           self.client = QdrantClient(host=host, port=port)
   ```

3. **Configure Streamlit Secrets**
   ```toml
   GEMINI_API_KEY = "your-gemini-api-key"
   SEMANTIC_SCHOLAR_API_KEY = "your-semantic-scholar-api-key"
   QDRANT_URL = "https://your-cluster.qdrant.io"
   QDRANT_API_KEY = "your-qdrant-api-key"
   ```

4. **Deploy** following the same steps as Option 1

---

## Required API Keys

### Google Gemini API Key (Required)
1. Go to https://ai.google.dev/
2. Click "Get API key in Google AI Studio"
3. Create a new API key

### Semantic Scholar API Key (Optional but Recommended)
1. Go to https://www.semanticscholar.org/product/api
2. Request an API key
3. Helps avoid rate limits on external paper searches

---

## Testing Your Deployment

After deployment:

1. Visit your Streamlit Cloud URL
2. Try the app with **"External papers only"** mode
3. Test query: "reduce unemployment women Bogota Colombia"
4. Verify it retrieves papers and generates a memo

---

## Monitoring and Logs

- View logs in Streamlit Cloud dashboard
- Monitor API usage for Gemini and Semantic Scholar
- Check for rate limiting issues

---

## Troubleshooting

### "Qdrant connection failed"
- Make sure you're using "External papers only" mode
- Or set up cloud Qdrant (Option 2)

### "API key not found"
- Verify secrets are correctly set in Streamlit Cloud settings
- Keys should be in `GEMINI_API_KEY` and `SEMANTIC_SCHOLAR_API_KEY`

### Rate limit errors
- Add Semantic Scholar API key to increase limits
- Reduce similarity threshold if needed
- Wait before retrying queries

---

## Cost Considerations

- **Streamlit Cloud**: Free tier available (limited to 1 app, 1GB RAM)
- **Gemini API**: Free tier: 15 requests/minute, 1500 requests/day
- **Semantic Scholar API**: Free without key (lower limits); higher with free API key
- **Qdrant Cloud**: Free tier available (1GB storage)
