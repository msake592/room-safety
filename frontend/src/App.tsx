import { useEffect, useState } from 'react';
import { AnalysisResult } from './components/AnalysisResult';
import { ImageUploader } from './components/ImageUploader';
import { AnalysisApiError, analyzeImage, isApiConfigured } from './services/analysisApi';
import type { AnalysisError, AnalysisResponse, AnalysisStatus, SelectedImage } from './types/analysis';
import './styles.css';

export default function App() {
  const [selectedImage, setSelectedImage] = useState<SelectedImage | null>(null);
  const [analysisStatus, setAnalysisStatus] = useState<AnalysisStatus>('idle');
  const [analysisResult, setAnalysisResult] = useState<AnalysisResponse | null>(null);
  const [analysisError, setAnalysisError] = useState<AnalysisError | null>(null);

  useEffect(() => {
    return () => {
      if (selectedImage) {
        URL.revokeObjectURL(selectedImage.previewUrl);
      }
    };
  }, [selectedImage]);

  function handleImageSelected(file: File) {
    setSelectedImage((currentImage) => {
      if (currentImage) {
        URL.revokeObjectURL(currentImage.previewUrl);
      }

      return {
        file,
        previewUrl: URL.createObjectURL(file),
      };
    });
    setAnalysisStatus('ready');
    setAnalysisResult(null);
    setAnalysisError(null);
  }

  async function handleAnalyzeClick() {
    if (!selectedImage) {
      setAnalysisStatus('error');
      setAnalysisError({
        code: 'NO_IMAGE_SELECTED',
        message: 'Upload a room image before starting analysis.',
      });
      return;
    }

    setAnalysisStatus('loading');
    setAnalysisError(null);

    try {
      const result = await analyzeImage(selectedImage.file);
      setAnalysisResult(result);
      setAnalysisStatus('success');
    } catch (error) {
      setAnalysisStatus('error');
      setAnalysisResult(null);
      setAnalysisError(
        error instanceof AnalysisApiError
          ? { code: error.code, message: error.message }
          : {
              code: 'BACKEND_UNREACHABLE',
              message: 'The image could not be analyzed. Please try again.',
            },
      );
    }
  }

  const hasImage = Boolean(selectedImage);
  const apiConfigured = isApiConfigured();

  return (
    <main className="app-shell">
      <section className="hero">
        <div>
          <p className="eyebrow">Child-safety image review</p>
          <h1>Room Safety</h1>
          <p className="hero-copy">
            Analyze room images for potential child-safety risks, then review hazards
            and recommendations in one focused workspace.
          </p>
        </div>
        <div className="status-strip" aria-label="Integration status">
          <span className={apiConfigured ? 'status-dot is-ready' : 'status-dot'} />
          <span>{apiConfigured ? 'API URL configured' : 'API URL not configured'}</span>
        </div>
      </section>

      <div className="workspace">
        <ImageUploader selectedImage={selectedImage} onImageSelected={handleImageSelected} />

        <div className="action-column">
          <button
            className="primary-action"
            type="button"
            disabled={!hasImage || analysisStatus === 'loading'}
            onClick={handleAnalyzeClick}
          >
            {analysisStatus === 'loading' ? 'Analyzing...' : 'Analyze Room'}
          </button>
          <p className="action-note">
            {analysisStatus === 'loading'
              ? 'Analysis can take a little while while the backend runs detection and segmentation.'
              : 'JPEG and PNG images are sent to the configured FastAPI backend.'}
          </p>
          <AnalysisResult
            status={analysisStatus}
            hasImage={hasImage}
            result={analysisResult}
            error={analysisError}
          />
        </div>
      </div>
    </main>
  );
}
