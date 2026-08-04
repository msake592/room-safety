export type SelectedImage = {
  file: File;
  previewUrl: string;
};

export type Detection = {
  label: string;
  raw_label?: string | null;
  canonical_label?: string | null;
  display_label?: string | null;
  score: number;
  box: number[];
};

export type Risk = Detection & {
  rule_id?: string | null;
  risk_level?: string | null;
  risk_score?: number | null;
  reason?: string | null;
  recommendation?: string | null;
  target_group?: string | null;
};

export type AnalysisResponse = {
  image_name: string;
  image_path?: string | null;
  detection_count: number;
  detections: Detection[];
  risk_count?: number | null;
  risks: Risk[];
  result_image_path: string;
  result_image_url?: string | null;
  risk_analysis_path?: string | null;
  boxed_image_path?: string | null;
};

export type AnalysisErrorCode =
  | 'NO_IMAGE_SELECTED'
  | 'API_NOT_CONFIGURED'
  | 'UNSUPPORTED_IMAGE'
  | 'BACKEND_UNREACHABLE'
  | 'BACKEND_4XX'
  | 'BACKEND_5XX'
  | 'MALFORMED_RESPONSE'
  | 'RESULT_IMAGE_UNAVAILABLE';

export type AnalysisError = {
  code: AnalysisErrorCode;
  message: string;
};

export type AnalysisStatus = 'idle' | 'ready' | 'loading' | 'success' | 'error';
