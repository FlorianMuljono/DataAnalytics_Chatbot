# Deployment Guide for Data Analytics Assistant

This guide will walk you through deploying your Data Analytics Assistant application to Streamlit Cloud, which provides free hosting for Streamlit applications.

## Prerequisite

1. GitHub account
2. OpenAI API key

## Deployment Steps to Streamlit Cloud

### 1. Prepare Your Code Repository

1. Download all code files from this Replit
2. Create a new GitHub repository (if not done already)
3. Make sure your repository includes:
   - All Python files (app.py, conversation.py, data_processor.py, utils.py)
   - requirements-deploy.txt (rename it to requirements.txt in your repository)
   - README.md
   - .streamlit/config.toml (create this if not present)

4. Create a .streamlit directory and config.toml file to ensure the application works correctly:
   ```bash
   mkdir -p .streamlit
   ```

5. Create a file at `.streamlit/config.toml` with this content:
   ```toml
   [server]
   headless = true
   enableCORS = false
   enableXsrfProtection = false
   ```

### 2. Push to GitHub

1. Initialize your local Git repository (if not done already):
   ```bash
   git init
   ```

2. Add all files:
   ```bash
   git add .
   ```

3. Commit changes:
   ```bash
   git commit -m "Initial commit"
   ```

4. Link to your GitHub repository (replace with your actual repository URL):
   ```bash
   git remote add origin https://github.com/yourusername/data-analytics-assistant.git
   ```

5. Push your code:
   ```bash
   git push -u origin main
   ```

### 3. Deploy on Streamlit Cloud

1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository, branch (main), and the file path to your app (app.py)
5. Click "Advanced settings"
6. Add your OpenAI API key as a secret with the name `OPENAI_API_KEY`
7. Click "Deploy"
8. Wait for deployment to complete

Once deployed, Streamlit Cloud will provide a URL where your app is accessible. The great advantage is that file uploads will work correctly in this environment!

## Local Development

To run the app locally:

1. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Set your OpenAI API key:
   ```bash
   # On Windows
   set OPENAI_API_KEY=your_api_key_here
   
   # On Mac/Linux
   export OPENAI_API_KEY=your_api_key_here
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Troubleshooting

- **Import errors**: Make sure all packages are installed correctly
- **API key errors**: Verify your OpenAI API key is correctly set
- **File upload issues**: Check that you have a recent version of Streamlit
- **Deployment fails**: Check Streamlit Cloud logs for details

## Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-cloud)
- [OpenAI API Documentation](https://platform.openai.com/docs/)