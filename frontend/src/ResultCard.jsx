import React from 'react';

const ResultCard = ({ result }) => {
  if (!result) return null;

  const { label, confidence, probabilities } = result;
  const isPositive = label === 'positive';
  const percentage = (confidence * 100).toFixed(1);

  return (
    <div className="glass-card result-container">
      <div className="result-header">
        <div>
          <p className="prob-label">Prediction</p>
          <div className={`result-label ${label}`}>
            {isPositive ? (
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                <polyline points="22 4 12 14.01 9 11.01"></polyline>
              </svg>
            ) : (
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="15" y1="9" x2="9" y2="15"></line>
                <line x1="9" y1="9" x2="15" y2="15"></line>
              </svg>
            )}
            {label}
          </div>
        </div>
      </div>

      <div className="confidence-section">
        <div className="confidence-header">
          <span>Confidence Score</span>
          <span>{percentage}%</span>
        </div>
        <div className="progress-bar-bg">
          <div 
            className={`progress-bar-fill ${label}`} 
            style={{ width: `${percentage}%` }}
          />
        </div>
      </div>

      <div className="probabilities">
        <div className="prob-item">
          <span className="prob-label">Negative</span>
          <span className="prob-val">{(probabilities.negative * 100).toFixed(1)}%</span>
        </div>
        <div className="prob-item">
          <span className="prob-label">Positive</span>
          <span className="prob-val">{(probabilities.positive * 100).toFixed(1)}%</span>
        </div>
        <div className="prob-item">
          <span className="prob-label">Inference Time</span>
          <span className="prob-val">{result.inference_time_ms} ms</span>
        </div>
      </div>
    </div>
  );
};

export default ResultCard;
