# Deploy Cogaly to Azure Web App
# Usage: ./deploy.ps1 [ResourceGroupName] [AppName]
# Uses B1 (Basic) tier for better performance with ML libraries

param (
    [string]$ResourceGroup = "cogaly-rg",
    [string]$AppName = "cogaly-app",
    [string]$Location = "eastus"
)

Write-Host "🚀 Starting Cogaly Deployment to Azure..." -ForegroundColor Cyan
Write-Host "   Using B1 (Basic) tier for ML workload support" -ForegroundColor Yellow

# Check if Azure CLI is installed
if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Azure CLI (az) is not installed or not in PATH." -ForegroundColor Red
    Write-Host "   Please restart your terminal if you just installed it."
    exit 1
}

# Login check
Write-Host "🔑 Checking Azure login status..."
az account show > $null 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  Please login to Azure..." -ForegroundColor Yellow
    az login
}

# Delete old resources if they exist (to avoid conflicts)
Write-Host "🧹 Cleaning up old deployment (if exists)..."
az webapp delete --name $AppName --resource-group $ResourceGroup 2>$null
az appservice plan delete --name "$AppName-plan" --resource-group $ResourceGroup --yes 2>$null

# Create Resource Group (idempotent)
Write-Host "📦 Creating Resource Group: $ResourceGroup..."
az group create --name $ResourceGroup --location $Location

# Create App Service Plan with B1 (Basic) tier
Write-Host "🏗️  Creating App Service Plan (B1 Basic Tier - ~`$13/month)..."
az appservice plan create --name "$AppName-plan" --resource-group $ResourceGroup --sku B1 --is-linux

# Create Web App
Write-Host "🌐 Creating Web App: $AppName..."
az webapp create --resource-group $ResourceGroup --plan "$AppName-plan" --name $AppName --runtime "PYTHON:3.10"

# Configure Startup Command
Write-Host "⚙️  Configuring startup command..."
az webapp config set --resource-group $ResourceGroup --name $AppName --startup-file 'python -m uvicorn app.app:app --host 0.0.0.0 --port $PORT'

# Enable Always On (keeps app warm, faster responses)
Write-Host "🔥 Enabling Always On..."
az webapp config set --resource-group $ResourceGroup --name $AppName --always-on true

# Deploy Code using zip deploy
Write-Host "📤 Deploying code (this may take a few minutes)..."
az webapp up --resource-group $ResourceGroup --name $AppName --sku B1

# Restart to apply all settings
Write-Host "🔄 Restarting app to apply settings..."
az webapp restart --resource-group $ResourceGroup --name $AppName

Write-Host ""
Write-Host "✅ Deployment Complete!" -ForegroundColor Green
Write-Host "🌍 Your app is live at: https://$AppName.azurewebsites.net" -ForegroundColor Cyan
Write-Host ""
Write-Host "💰 Estimated cost: ~`$13/month (B1 Basic tier)" -ForegroundColor Yellow
Write-Host "   You have `$90 in credits, so this will last ~7 months" -ForegroundColor Yellow
