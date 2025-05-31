import kagglehub

# Download latest version
path = kagglehub.dataset_download("alaakhaled/conll003-englishversion")

print("Path to dataset files:", path)