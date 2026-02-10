import { useState } from "react";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();

    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:5000/predict", {
        method: "POST",
        body: formData
      });

      const data = await res.json();
      setResult(data);

    } catch (error) {
      console.error("Error:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <div className="card">
        <h1>🤖 CropAI Agent</h1>
        <p>Upload crop image. I will analyze it.</p>

        <input type="file" onChange={(e) => setFile(e.target.files[0])} />
        <button onClick={handleSubmit}>Analyze</button>

        {loading && <p className="thinking">🧠 Analyzing...</p>}

        {result && (
          <div className="result">
            <h2>{result.crop}</h2>
            <p>Confidence: {result.confidence}%</p>
            <img src={result.image} alt="crop" />
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
