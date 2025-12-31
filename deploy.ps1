param (
    [string]$ResourceGroup = "cogaly-rg",
    [string]$AppName = "cogaly-app",
    [string]$Location = "eastus"
)

Write-Host "🚀 Updating Cogaly Deployment to Python 3.11..." -ForegroundColor Cyan

# Login check
az account show > $null 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  Please login to Azure..." -ForegroundColor Yellow
    az login
}

# Update Runtime to Python 3.11 (Managed via Portal or Setup Script)
Write-Host "🐍 Runtime: Python 3.11 (Ensure this is set in Azure Portal)"


# Deploy Code using zip deploy
Write-Host "📤 Deploying code (this may take a few minutes)..."
# Note: az webapp up might reset some configs, so we set them again after
az webapp up --resource-group $ResourceGroup --name $AppName --sku B1

# Configure Startup Command
Write-Host "⚙️  Configuring startup command..."
az webapp config set --resource-group $ResourceGroup --name $AppName --startup-file 'python -m uvicorn app.app:app --host 0.0.0.0 --port $PORT'

# Enable Always On
Write-Host "🔥 Enabling Always On..."
az webapp config set --resource-group $ResourceGroup --name $AppName --always-on true

# Restart to apply all settings
Write-Host "🔄 Restarting app to apply settings..."
az webapp restart --resource-group $ResourceGroup --name $AppName

Write-Host ""
Write-Host "✅ Deployment Complete!" -ForegroundColor Green
Write-Host "🌍 Your app is live at: https://$AppName.azurewebsites.net" -ForegroundColor Cyan

