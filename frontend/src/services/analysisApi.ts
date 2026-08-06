import type { AnalysisErrorCode, AnalysisResponse } from '../types/analysis';

const rawApiBaseUrl = import.meta.env.VITE_API_BASE_URL;

export const apiConfig = {
  baseUrl: rawApiBaseUrl ? rawApiBaseUrl.replace(/\/$/, '') : '',
};

export function getAnalyzeEndpoint(): string {
  return `${apiConfig.baseUrl}/analyze`;
}

export function isApiConfigured(): boolean {
  return Boolean(apiConfig.baseUrl);
}

export class AnalysisApiError extends Error {
  code: AnalysisErrorCode;
  status?: number;

  constructor(message: string, code: AnalysisErrorCode, status?: number) {
    super(message);
    this.name = 'AnalysisApiError';
    this.code = code;
    this.status = status;
  }
}

type BackendErrorResponse = {
  error?: {
    code?: string;
    message?: string;
  };
};

function isAnalysisResponse(value: unknown): value is AnalysisResponse {
  if (!value || typeof value !== 'object') {
    return false;
  }

  const candidate = value as Partial<AnalysisResponse>;

  return (
    typeof candidate.image_name === 'string' &&
    typeof candidate.detection_count === 'number' &&
    Array.isArray(candidate.detections) &&
    Array.isArray(candidate.risks) &&
    typeof candidate.result_image_path === 'string'
  );
}

function getErrorMessageForStatus(status: number, fallback?: string): string {
  if (fallback) {
    return fallback;
  }

  if (status >= 500) {
    return 'The analysis service had a problem. Please try again later.';
  }

  return 'The image could not be analyzed. Please check the file and try again.';
}

function getErrorCodeForStatus(status: number): AnalysisErrorCode {
  return status >= 500 ? 'BACKEND_5XX' : 'BACKEND_4XX';
}

export function getAbsoluteApiUrl(pathOrUrl: string): string {
  if (/^https?:\/\//i.test(pathOrUrl)) {
    return pathOrUrl;
  }

  return `${apiConfig.baseUrl}${pathOrUrl.startsWith('/') ? '' : '/'}${pathOrUrl}`;
}

export async function analyzeImage(image: File): Promise<AnalysisResponse> {
  if (!isApiConfigured()) {
    throw new AnalysisApiError(
      'The API URL is not configured. Set VITE_API_BASE_URL and restart the frontend.',
      'API_NOT_CONFIGURED',
    );
  }

  if (!['image/jpeg', 'image/png'].includes(image.type)) {
    throw new AnalysisApiError(
      'Only JPEG and PNG images are supported.',
      'UNSUPPORTED_IMAGE',
    );
  }

  const formData = new FormData();
  formData.append('image', image);

  let response: Response;

  try {
    response = await fetch(getAnalyzeEndpoint(), {
      method: 'POST',
      body: formData,
    });
  } catch (error) {
    throw new AnalysisApiError(
      'The backend could not be reached, or the browser blocked the request. Check that the backend is running and CORS allows this frontend URL.',
      'BACKEND_UNREACHABLE',
    );
  }

  let responseJson: unknown;

  try {
    responseJson = await response.json();
  } catch (error) {
    throw new AnalysisApiError(
      'The backend returned a response the frontend could not read.',
      'MALFORMED_RESPONSE',
      response.status,
    );
  }

  if (!response.ok) {
    const backendError = responseJson as BackendErrorResponse;
    throw new AnalysisApiError(
      getErrorMessageForStatus(response.status, backendError.error?.message),
      getErrorCodeForStatus(response.status),
      response.status,
    );
  }

  if (!isAnalysisResponse(responseJson)) {
    throw new AnalysisApiError(
      'The backend returned analysis data in an unexpected format.',
      'MALFORMED_RESPONSE',
      response.status,
    );
  }

  return responseJson;
}
