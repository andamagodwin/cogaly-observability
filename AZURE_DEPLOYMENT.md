# Cogaly Azure Deployment Guide

## 🚀 Deploy to Azure Web App (Recommended)

### Prerequisites
- Azure account (free tier works!)
- Azure CLI installed (`winget install Microsoft.AzureCLI`)
- Docker Desktop (optional, for container deployment)

---

## Option 1: Azure App Service (Easiest)

### Step 1: Login to Azure
```bash
az login
```

### Step 2: Create Resource Group
```bash
az group create --name cogaly-rg --location eastus
```

### Step 3: Create App Service Plan
```bash
az appservice plan create \
    --name cogaly-plan \
    --resource-group cogaly-rg \
    --sku B1 \
    --is-linux
```

### Step 4: Create Web App
```bash
az webapp create \
    --resource-group cogaly-rg \
    --plan cogaly-plan \
    --name cogaly-app \
    --runtime "PYTHON:3.10"
```

### Step 5: Configure Startup Command
```bash
az webapp config set \
    --resource-group cogaly-rg \
    --name cogaly-app \
    --startup-file "cd app && uvicorn app:app --host 0.0.0.0 --port 8000"
```

### Step 6: Deploy Code
```bash
# From the project root directory
az webapp deploy \
    --resource-group cogaly-rg \
    --name cogaly-app \
    --src-path . \
    --type zip
```

### Step 7: Access Your App
Your app will be available at: `https://cogaly-app.azurewebsites.net`

---

## Option 2: Azure Container Instances (Docker)

### Step 1: Build Docker Image
```bash
docker build -t cogaly:latest .
```

### Step 2: Create Azure Container Registry
```bash
az acr create \
    --resource-group cogaly-rg \
    --name cogalyregistry \
    --sku Basic
```

### Step 3: Login to Registry
```bash
az acr login --name cogalyregistry
```

### Step 4: Tag and Push Image
```bash
docker tag cogaly:latest cogalyregistry.azurecr.io/cogaly:latest
docker push cogalyregistry.azurecr.io/cogaly:latest
```

### Step 5: Deploy Container
```bash
az container create \
    --resource-group cogaly-rg \
    --name cogaly-container \
    --image cogalyregistry.azurecr.io/cogaly:latest \
    --dns-name-label cogaly-demo \
    --ports 8000 \
    --registry-login-server cogalyregistry.azurecr.io \
    --registry-username $(az acr credential show --name cogalyregistry --query username -o tsv) \
    --registry-password $(az acr credential show --name cogalyregistry --query passwords[0].value -o tsv)
```

---

## Option 3: Azure Static Web Apps + Functions (Serverless)

For a fully serverless deployment, you can:
1. Deploy the frontend to Azure Static Web Apps
2. Deploy the API as an Azure Function

This is more complex but offers better scaling and lower costs for low traffic.

---

## 🧪 Test Locally First

Before deploying, test locally:

```bash
# Install dependencies
cd app
pip install -r requirements.txt

# Run the server
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Then open: http://localhost:8000

---

## 📊 Environment Variables (Optional)

For production, set these environment variables in Azure:

| Variable | Description |
|----------|-------------|
| `MODEL_DIR` | Path to model artifacts (default: `../model`) |
| `DEBUG` | Enable debug mode (default: `false`) |

Set via Azure CLI:
```bash
az webapp config appsettings set \
    --resource-group cogaly-rg \
    --name cogaly-app \
    --settings DEBUG=false
```

---

## 🔒 Security Considerations

For production:
1. Enable HTTPS only
2. Add authentication (Azure AD, API keys)
3. Set up CORS properly
4. Enable logging and monitoring
5. Add rate limiting

---

## 💰 Cost Estimation

| Resource | Tier | Monthly Cost |
|----------|------|--------------|
| App Service Plan | B1 | ~$13/month |
| Container Instance | 1 vCPU, 1.5 GB | ~$30/month |
| Static Web App | Free | $0 |

**Free tier options:**
- Azure App Service has a free F1 tier (limited)
- Students get $100 free credits

---

## 🆘 Troubleshooting

### App won't start
```bash
az webapp log tail --resource-group cogaly-rg --name cogaly-app
```

### Model not loading
Ensure all `.pkl` files are in the `model/` directory and included in deployment.

### CORS errors
The API is configured to allow all origins (`*`). For production, restrict this.

---

**Need help?** Open an issue on GitHub!
