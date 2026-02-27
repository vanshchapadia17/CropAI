import { useState, useRef, useEffect } from "react";
import "./App.css";

function App() {
  const [messages, setMessages] = useState([
    { role: "bot", text: "Hello! I'm CropAI, your smart agriculture assistant.\n\nHere's what I can do for you:\n🌱 Identify crops — upload a field image to detect Jute, Maize, Rice, Sugarcane, or Wheat\n🔬 Detect diseases — upload a leaf image to check for Rice or Sugarcane diseases\n📚 Answer farming questions — ask me anything about Rice or Sugarcane cultivation, pests, irrigation, nutrients, and more\n\nHow can I help you today?" }
  ]);
  const [input, setInput] = useState("");
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() && !file) return;

    // Add user message
    const userMsg = {
      role: "user",
      text: input,
      image: file ? URL.createObjectURL(file) : null
    };
    setMessages(prev => [...prev, userMsg]);

    const formData = new FormData();
    formData.append("message", input);
    if (file) formData.append("file", file);

    // Send chat history for conversation memory
    const historyForBackend = messages.map(msg => ({
      role: msg.role,
      text: msg.text || ""
    }));
    formData.append("chat_history", JSON.stringify(historyForBackend));

    setInput("");
    setFile(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
    setLoading(true);

    try {
      const res = await fetch("http://localhost:5000/chat", {
        method: "POST",
        body: formData
      });
      const data = await res.json();

      if (data.error) {
        setMessages(prev => [...prev, { role: "bot", text: `Error: ${data.error}` }]);
      } else {
        const botMsg = {
          role: "bot",
          text: data.response,
          image: data.image || null,
        };
        setMessages(prev => [...prev, botMsg]);
      }
    } catch (error) {
      setMessages(prev => [...prev, { role: "bot", text: "Something went wrong. Please try again." }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h1>🤖 CropAI Agent</h1>
        <p>Crop Identifier • Disease Detector • Agriculture Assistant</p>
      </div>

      <div className="chat-messages">
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            {msg.role === "bot" && <span className="avatar">🤖</span>}
            <div className="bubble">
              {msg.text && <p>{msg.text}</p>}
              {msg.image && <img src={msg.image} alt="uploaded" />}
            </div>
            {msg.role === "user" && <span className="avatar">👤</span>}
          </div>
        ))}
        {loading && (
          <div className="message bot">
            <span className="avatar">🤖</span>
            <div className="bubble typing">
              <span></span><span></span><span></span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input">
        {file && (
          <div className="file-preview">
            <img src={URL.createObjectURL(file)} alt="preview" />
            <button className="remove-file" onClick={() => { setFile(null); if (fileInputRef.current) fileInputRef.current.value = ""; }}>✕</button>
          </div>
        )}
        <div className="input-row">
          <button className="attach-btn" onClick={() => fileInputRef.current?.click()}>📎</button>
          <input
            type="file"
            ref={fileInputRef}
            hidden
            accept="image/*"
            onChange={(e) => setFile(e.target.files[0])}
          />
          <input
            type="text"
            placeholder="Ask about a crop or disease..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
          />
          <button className="send-btn" onClick={handleSend} disabled={loading}>
            ➤
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
