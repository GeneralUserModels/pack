#!/bin/bash

# Script to zip all JSON files ending with _result.json from nested session/chunks folders

# Set the output zip file name with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_ZIP="result_jsons_${TIMESTAMP}.zip"

echo "Searching for *_result.json files in logs folder..."
echo "Output will be saved as: $OUTPUT_ZIP"

# Find all files ending with _result.json in the logs directory and its subdirectories
# Then zip them into the output file
find logs -name "*_result.json" -type f | zip -@ "$OUTPUT_ZIP"

# Check if the zip was created successfully
if [ -f "$OUTPUT_ZIP" ]; then
    # Count the number of files in the zip
    FILE_COUNT=$(unzip -l "$OUTPUT_ZIP" | tail -1 | awk '{print $2}')
    echo "Successfully created $OUTPUT_ZIP"
    echo "Total files zipped: $FILE_COUNT"
    echo "File size: $(du -h "$OUTPUT_ZIP" | cut -f1)"
else
    echo "Error: Failed to create zip file"
    exit 1
fi
