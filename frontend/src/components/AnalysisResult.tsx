import { useEffect, useState } from 'react';
import { getAbsoluteApiUrl } from '../services/analysisApi';
import type {
  AnalysisError,
  AnalysisResponse,
  AnalysisStatus,
  Detection,
  Risk,
} from '../types/analysis';

type AnalysisResultProps = {
  status: AnalysisStatus;
  hasImage: boolean;
  result: AnalysisResponse | null;
  error: AnalysisError | null;
};

function formatScore(score: number): string {
  return `${Math.round(score * 100)}%`;
}

function formatRiskLevel(riskLevel?: string | null): string {
  if (!riskLevel) {
    return 'Risk';
  }

  return riskLevel.replace(/_/g, ' ');
}

function getHighestRiskLevel(risks: Risk[]): string {
  const ranking: Record<string, number> = {
    critical: 4,
    high: 3,
    medium: 2,
    low: 1,
  };

  return risks.reduce((highestLevel, risk) => {
    const currentLevel = risk.risk_level?.toLowerCase() ?? '';
    const highestScore = ranking[highestLevel] ?? 0;
    const currentScore = ranking[currentLevel] ?? 0;

    return currentScore > highestScore ? currentLevel : highestLevel;
  }, 'none');
}

function getUniqueRecommendations(risks: Risk[]): string[] {
  const recommendations = new Set<string>();

  risks.forEach((risk) => {
    if (risk.recommendation?.trim()) {
      recommendations.add(risk.recommendation.trim());
    }
  });

  return Array.from(recommendations);
}

function HazardCard({ risk }: { risk: Risk }) {
  const title = risk.display_label ?? risk.canonical_label ?? risk.label;

  return (
    <li className="hazard-card">
      <div className="hazard-card-header">
        <div>
          <span className="hazard-label">{title}</span>
          {risk.rule_id ? <span className="hazard-rule">{risk.rule_id}</span> : null}
        </div>
        <span className={`risk-level risk-level-${risk.risk_level ?? 'unknown'}`}>
          {formatRiskLevel(risk.risk_level)}
        </span>
      </div>
      <dl className="hazard-metrics">
        <div>
          <dt>Detection score</dt>
          <dd>{formatScore(risk.score)}</dd>
        </div>
        {risk.risk_score != null ? (
          <div>
            <dt>Risk score</dt>
            <dd>{risk.risk_score}</dd>
          </div>
        ) : null}
        {risk.target_group ? (
          <div>
            <dt>Target group</dt>
            <dd>{risk.target_group}</dd>
          </div>
        ) : null}
      </dl>
      {risk.reason ? <p className="hazard-reason">{risk.reason}</p> : null}
    </li>
  );
}

function TechnicalDetections({ detections }: { detections: Detection[] }) {
  if (detections.length === 0) {
    return null;
  }

  return (
    <details className="technical-detections">
      <summary>Technical detections</summary>
      <ul>
        {detections.map((detection, index) => (
          <li key={`${detection.label}-${index}`}>
            <span>
              {detection.display_label ?? detection.canonical_label ?? detection.label}
              {detection.raw_label && detection.raw_label !== detection.display_label ? (
                <small>Raw: {detection.raw_label}</small>
              ) : null}
            </span>
            <span>{formatScore(detection.score)}</span>
          </li>
        ))}
      </ul>
    </details>
  );
}

function SafetySummary({ result }: { result: AnalysisResponse }) {
  const riskCount = result.risk_count ?? result.risks.length;
  const highestRiskLevel = getHighestRiskLevel(result.risks);

  return (
    <section className="report-section safety-summary" aria-labelledby="safety-summary-title">
      <div className="report-section-heading">
        <p className="eyebrow">Safety summary</p>
        <h3 id="safety-summary-title">
          {riskCount > 0
            ? `${riskCount} configured hazard${riskCount === 1 ? '' : 's'} found`
            : 'No configured hazards found'}
        </h3>
      </div>
      <div className="summary-grid">
        <div>
          <span className="summary-value">{riskCount}</span>
          <span className="summary-label">Hazards</span>
        </div>
        <div>
          <span className="summary-value summary-value-text">
            {formatRiskLevel(highestRiskLevel)}
          </span>
          <span className="summary-label">Highest risk</span>
        </div>
        <div>
          <span className="summary-value">{result.detection_count}</span>
          <span className="summary-label">Technical detections</span>
        </div>
      </div>
    </section>
  );
}

function HazardCards({ risks }: { risks: Risk[] }) {
  return (
    <section className="report-section" aria-labelledby="hazards-title">
      <div className="report-section-heading">
        <p className="eyebrow">Hazard cards</p>
        <h3 id="hazards-title">Matched safety risks</h3>
      </div>
      {risks.length > 0 ? (
        <ul className="hazard-list">
          {risks.map((risk, index) => (
            <HazardCard key={`${risk.label}-${risk.rule_id ?? index}`} risk={risk} />
          ))}
        </ul>
      ) : (
        <p className="empty-result">No configured risks were detected.</p>
      )}
    </section>
  );
}

function Recommendations({ risks }: { risks: Risk[] }) {
  const recommendations = getUniqueRecommendations(risks);

  return (
    <section className="report-section" aria-labelledby="recommendations-title">
      <div className="report-section-heading">
        <p className="eyebrow">Recommendations</p>
        <h3 id="recommendations-title">Next actions</h3>
      </div>
      {recommendations.length > 0 ? (
        <ol className="recommendation-list">
          {recommendations.map((recommendation) => (
            <li key={recommendation}>{recommendation}</li>
          ))}
        </ol>
      ) : (
        <p className="empty-result">No safety recommendations are needed for the matched rules.</p>
      )}
    </section>
  );
}

export function AnalysisResult({ status, hasImage, result, error }: AnalysisResultProps) {
  const [isResultImageUnavailable, setIsResultImageUnavailable] = useState(false);
  const resultImageUrl = result?.result_image_url
    ? getAbsoluteApiUrl(result.result_image_url)
    : null;

  useEffect(() => {
    setIsResultImageUnavailable(false);
  }, [resultImageUrl]);

  return (
    <section className="panel results-panel" aria-labelledby="results-title">
      <div className="section-heading">
        <p className="eyebrow">Analysis</p>
        <h2 id="results-title">Safety review</h2>
      </div>

      {status === 'success' && result ? (
        <div className="analysis-output">
          {resultImageUrl ? (
            isResultImageUnavailable ? (
              <p className="message message-error">Result image is unavailable.</p>
            ) : (
              <img
                className="result-image"
                src={resultImageUrl}
                alt="Masked room safety analysis result"
                onLoad={() => setIsResultImageUnavailable(false)}
                onError={() => setIsResultImageUnavailable(true)}
              />
            )
          ) : (
            <p className="message message-error">Result image is unavailable.</p>
          )}

          <SafetySummary result={result} />
          <HazardCards risks={result.risks} />
          <Recommendations risks={result.risks} />
          <TechnicalDetections detections={result.detections} />
        </div>
      ) : (
        <div className="results-placeholder">
          {status === 'loading' ? (
            <>
              <p className="result-title">Analyzing image...</p>
              <p>The backend is detecting objects, evaluating risk rules, and creating the masked result image.</p>
            </>
          ) : status === 'error' ? (
            <>
              <p className="result-title">Analysis could not be completed.</p>
              <p>{error?.message ?? 'Please try again.'}</p>
            </>
          ) : hasImage ? (
            <>
              <p className="result-title">Ready to analyze.</p>
              <p>Start analysis to display detected hazards, risk notes, and the masked result image here.</p>
            </>
          ) : (
            <>
              <p className="result-title">No image selected.</p>
              <p>Upload a room image to prepare it for analysis.</p>
            </>
          )}
        </div>
      )}
    </section>
  );
}
