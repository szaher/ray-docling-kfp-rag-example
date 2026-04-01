#!/usr/bin/env python3
"""Re-upload the updated RAG pipeline with increased VARCHAR limit.

Run this to update an existing pipeline with the new schema.
"""
import kfp
import subprocess

# Configuration
NAMESPACE = "ray-docling"
PIPELINE_FILE = "rag_multistep_pipeline.yaml"
PIPELINE_NAME = "rag-multi-step-pipeline"

# Get DSPA route
result = subprocess.run(
    ["oc", "get", "route", "ds-pipeline-dspa", "-n", NAMESPACE, "-o", "jsonpath={.spec.host}"],
    capture_output=True,
    text=True,
    check=True,
)
dspa_host = result.stdout.strip()
dspa_url = f"https://{dspa_host}"

# Get auth token
result = subprocess.run(
    ["oc", "whoami", "-t"],
    capture_output=True,
    text=True,
    check=True,
)
token = result.stdout.strip()

# Create KFP client
client = kfp.Client(
    host=dspa_url,
    existing_token=token,
    verify_ssl=False,
)

# Check if pipeline exists
try:
    pipelines = client.list_pipelines(filter=f'name="{PIPELINE_NAME}"')
    if pipelines.pipelines:
        # Pipeline exists, get its ID
        pipeline_id = pipelines.pipelines[0].pipeline_id
        print(f"Found existing pipeline: {PIPELINE_NAME} (ID: {pipeline_id})")

        # Upload new version
        print(f"Uploading new version from {PIPELINE_FILE}...")
        pipeline = client.upload_pipeline_version(
            pipeline_package_path=PIPELINE_FILE,
            pipeline_version_name=f"v2-varchar-32k",
            pipeline_id=pipeline_id,
        )
        print(f"✅ New pipeline version uploaded: {pipeline.pipeline_version_id}")
        print(f"   Schema now supports VARCHAR up to 32,768 characters")
    else:
        # Pipeline doesn't exist, create new
        print(f"Pipeline '{PIPELINE_NAME}' not found, creating new...")
        pipeline = client.upload_pipeline(
            pipeline_package_path=PIPELINE_FILE,
            pipeline_name=PIPELINE_NAME,
        )
        print(f"✅ Pipeline created: {pipeline.pipeline_id}")

except Exception as e:
    print(f"Error: {e}")
    print("\nYou can also manually re-upload via the RHOAI dashboard:")
    print("1. Go to Data Science Pipelines → Pipelines")
    print(f"2. Find '{PIPELINE_NAME}'")
    print("3. Click 'Upload new version'")
    print(f"4. Upload {PIPELINE_FILE}")
