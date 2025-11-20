from typing import Optional, Any, Dict, List, Union
import json
from pathlib import Path
from google.cloud import storage
from google.cloud import bigquery
from label.clients.client import VLMClient, CAPTION_SCHEMA


class BigQueryResponse:
    def __init__(self, result_row):
        self.result_row = result_row
        self._json = None

    @property
    def text(self) -> str:
        # BigQuery returns the result in ml_generate_text_llm_result column
        return self.result_row.ml_generate_text_llm_result

    @property
    def json(self):
        if self._json is None:
            self._json = json.loads(self.text)
        return self._json


class BigQueryClient(VLMClient):
    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        model_name: str,
        bucket_name: str,
        gcs_prefix: str = "video_chunks",
        object_table_location: str = "us",  # e.g., "us.screenomics-gemini"
        temperature: float = 0.0,
    ):
        """
        Initialize BigQuery client for ML.GENERATE_TEXT with video analysis.
        
        Args:
            project_id: GCP project ID
            dataset_id: BigQuery dataset containing the remote model
            model_name: Name of the Gemini remote model in BigQuery
            bucket_name: GCS bucket name for uploading videos
            gcs_prefix: Prefix/folder path in GCS bucket
            object_table_location: Object table location (e.g., "us.screenomics-gemini")
            temperature: Model temperature parameter
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.model_name = model_name
        self.bucket_name = bucket_name
        self.gcs_prefix = gcs_prefix
        self.object_table_location = object_table_location
        self.temperature = temperature
        
        self.storage_client = storage.Client(project=project_id)
        self.bq_client = bigquery.Client(project=project_id)

    def upload_file(self, path: str) -> str:
        """
        Upload file to GCS and return the GCS URI.
        
        Args:
            path: Local file path
            
        Returns:
            GCS URI (gs://bucket/path/to/file)
        """
        file_path = Path(path)
        destination_blob_name = f"{self.gcs_prefix}/{file_path.name}"
        
        bucket = self.storage_client.bucket(self.bucket_name)
        blob = bucket.blob(destination_blob_name)
        
        # Upload the file
        blob.upload_from_filename(path)
        
        gcs_uri = f"gs://{self.bucket_name}/{destination_blob_name}"
        print(f"Uploaded {path} to {gcs_uri}")
        
        return gcs_uri

    def generate(
        self, 
        prompt: str, 
        file_descriptor: Optional[Any] = None,
        schema: Optional[Dict] = None
    ) -> BigQueryResponse:
        """
        Generate text using BigQuery ML.GENERATE_TEXT with video from GCS.
        
        Args:
            prompt: Text prompt for the model
            file_descriptor: GCS URI returned from upload_file()
            schema: JSON schema for response (currently ignored, uses FLATTEN_JSON_OUTPUT)
            
        Returns:
            BigQueryResponse object with the model's response
        """
        if not file_descriptor:
            raise ValueError("file_descriptor (GCS URI) is required")
        
        gcs_uri = file_descriptor
        
        # Construct the BigQuery SQL query
        # Using OBJ.FETCH_METADATA to reference the video file in GCS
        query = f"""
        SELECT ml_generate_text_llm_result
        FROM ML.GENERATE_TEXT(
          MODEL `{self.project_id}.{self.dataset_id}.{self.model_name}`,
          (
            SELECT 
              STRUCT(
                @prompt, 
                OBJ.FETCH_METADATA(OBJ.MAKE_REF(@gcs_uri, @object_table_location))
              ) AS prompt 
          ),
          STRUCT(
            @temperature AS temperature,
            TRUE AS FLATTEN_JSON_OUTPUT
          )
        )
        """
        
        # Configure query with parameters
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("prompt", "STRING", prompt),
                bigquery.ScalarQueryParameter("gcs_uri", "STRING", gcs_uri),
                bigquery.ScalarQueryParameter("object_table_location", "STRING", self.object_table_location),
                bigquery.ScalarQueryParameter("temperature", "FLOAT64", self.temperature),
            ]
        )
        
        # Execute the query
        query_job = self.bq_client.query(query, job_config=job_config)
        
        # Wait for the query to complete and get results
        results = query_job.result()
        
        # Get the first (and only) row
        for row in results:
            return BigQueryResponse(row)
        
        raise RuntimeError("No results returned from BigQuery")