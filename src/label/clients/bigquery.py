from typing import Optional, Any, Dict
import json
from pathlib import Path

from google.cloud import storage
from google.cloud import bigquery

from label.clients.client import VLMClient, CAPTION_SCHEMA  # still imported, but we don't rely on it by default


class BigQueryResponse:
    def __init__(self, result_row):
        # get rid of the JSON scaffolds-
        self.result_row = result_row.replace("```json\n", "").replace("\n```", "")
        self._json = None

    @property
    def text(self) -> str:
        # BigQuery returns the result in ml_generate_text_llm_result column
        return self.result_row

    @property
    def json(self):
        if self._json is None:
            self._json = json.loads(self.text)
        return self._json


class BigQueryClient(VLMClient):
    def __init__(
        self,
        model_name: str,
        bucket_name: str,
        gcs_prefix: str = "video_chunks",
        object_table_location: str = "us",  # e.g., "us.screenomics-gemini"
        temperature: float = 0.0,
        max_output_tokens: int = 65535,
        project_id: Optional[str] = None,
    ):
        """
        Initialize BigQuery client for ML.GENERATE_TEXT with video analysis.

        Args:
            model_name: Full BigQuery model reference (e.g., "dataset.model" or "project.dataset.model")
            bucket_name: GCS bucket name for uploading videos
            gcs_prefix: Prefix/folder path in GCS bucket
            object_table_location: Object table location (e.g., "us.screenomics-gemini")
            temperature: Model temperature parameter
            max_output_tokens: Maximum number of tokens in the generated response
            project_id: Optional GCP project ID (if not provided, uses default credentials)
        """
        self.model_name = model_name
        self.bucket_name = bucket_name
        self.gcs_prefix = gcs_prefix
        self.object_table_location = object_table_location
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

        self.storage_client = storage.Client(project=project_id)
        self.bq_client = bigquery.Client(project=project_id)

    @staticmethod
    def _escape_for_bq_single_quoted_string(s: str) -> str:
        r"""
        Escape a Python string for use in a BigQuery single-quoted string literal.

        BigQuery-style escaping:
        - Backslash:    \  -> \\
        - Newline:      actual newline -> \n  (two chars)
        - Carriage ret: actual \r      -> \r
        - Single quote: '  -> \'
        """
        # Escape backslashes first so we don't re-escape ones we add later
        s = s.replace("\\", "\\\\")
        # Encode newlines and carriage returns as literal escape sequences
        s = s.replace("\r", "\\r")
        s = s.replace("\n", "\\n")
        # Escape single quotes for BigQuery
        s = s.replace("'", "\\'")
        return s

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
        schema: Optional[Dict] = None,
    ) -> BigQueryResponse:
        if not file_descriptor:
            raise ValueError("file_descriptor (GCS URI) is required")

        gcs_uri = file_descriptor

        # Escape everything that will go inside single-quoted SQL string literals
        escaped_prompt = self._escape_for_bq_single_quoted_string(prompt)
        escaped_gcs_uri = self._escape_for_bq_single_quoted_string(gcs_uri)
        escaped_location = self._escape_for_bq_single_quoted_string(
            self.object_table_location
        )

        # couldn't figure out how to pass a JSON schema.... :')
        query = f"""
        SELECT *
        FROM AI.GENERATE_TEXT(
          MODEL `{self.model_name}`,
          (
            SELECT STRUCT(
              '{escaped_prompt}' AS prompt,
              OBJ.FETCH_METADATA(
                OBJ.MAKE_REF('{escaped_gcs_uri}', '{escaped_location}')
              ) AS media
            ) AS prompt
          ),
          STRUCT(
            {self.max_output_tokens} AS max_output_tokens,
            {float(self.temperature)} AS temperature
          )
        )
        """

        job_config = bigquery.QueryJobConfig()

        print("=" * 80)
        print("BigQuery AI.GENERATE_TEXT Query:")
        print("=" * 80)
        print(query)
        print("=" * 80)

        query_job = self.bq_client.query(query, job_config=job_config)
        results = query_job.result()

        for row in results:
            print(row[0])
            return BigQueryResponse(row[0])

        raise RuntimeError("No results returned from BigQuery")
