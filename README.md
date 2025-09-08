# Garment-Attribute-Extractor

To run the services, Docker and WSL is required

Build the base image with:

docker compose build base --no-cache --progress=plain

Then build the other services and run them:

docker compose up --build --force-recreate

The services can be called for a set of image URLs (4 image URLs are needed) with a command such as:

Invoke-RestMethod -Uri "http://localhost:8000/v1/items/analyze" `
>>   -Method POST `
>>   -Headers @{ "Content-Type" = "application/json" } `
>>   -Body '{"image_urls":["https://AronRedivivum.github.io/Home-Assignment/Images/2025_08_2009_01_320011.JPG", "https://AronRedivivum.github.io/Home-Assignment/Images/2025_08_2009_01_490012.JPG", "https://AronRedivivum.github.io/Home-Assignment/Images/2025_08_2009_02_020040.JPG", "https://AronRedivivum.github.io/Home-Assignment/Images/2025_08_2009_02_150041.JPG"]}' `
>>   -OutFile "C:\response.json"

(Ran from Powershell)

To be continued