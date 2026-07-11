import { useState } from 'react'
import ResultCard from './ResultCard'

function App() {
  const [text, setText] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')

  const API_URL = "https://sentiment-classification-using-bert.onrender.com/predict"

  const handleAnalyze = async () => {
    if (!text.trim()) return

    setIsLoading(true)
    setError('')
    setResult(null)

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: text.trim() })
      })

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`)
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      console.error(err)
      setError('Failed to reach the API. Please ensure the backend is running and CORS is enabled.')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="app-container">
      <div className="header">
        <h1>BERT Sentiment Analysis</h1>
        <p>AI-powered sentiment classification using a fine-tuned Hugging Face transformer.</p>
      </div>

      <div className="glass-card">
        <div className="input-section">
          <textarea 
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Type or paste a sentence here to analyze its sentiment..."
            disabled={isLoading}
          />
          <button 
            className="analyze-btn" 
            onClick={handleAnalyze}
            disabled={isLoading || !text.trim()}
          >
            {isLoading ? (
              <><div className="spinner" /> Analyzing...</>
            ) : (
              'Analyze Sentiment'
            )}
          </button>
        </div>
      </div>

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      <ResultCard result={result} />
    </div>
  )
}

export default App
