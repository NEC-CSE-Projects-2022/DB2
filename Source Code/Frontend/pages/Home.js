import React from "react";
import "./Home.css";
import { Link } from "react-router-dom";

function Home() {
  return (
    <div>
      {/* Top Section (Left + Right) */}
      <div className="home">
        {/* Left Section */}
        <div className="home-left">
          <h4 className="tagline">TWITTER • BOT DETECTION • AI</h4>
          <h1 className="title">
            Detecting Bots on Twitter: <br />
            Hybrid Deep Learning Approach
          </h1>
          <p className="subtitle">
            A modern homepage for your project. It highlights the pipeline,
            explains models, and organizes references—fully responsive with
            glowing cards.
          </p>

          <div className="buttons">
            <Link to="/prediction" className="btn primary">
              Try Prediction
            </Link>
            <Link to="/models" className="btn secondary">
              Explore Models
            </Link>
          </div>
        </div>

        {/* Right Section (Cards) */}
        <div className="home-right">
          <div className="card">
            <h3>Data Collection</h3>
            <p>
              Gather user profiles, tweets, and metadata from TwiBot-20 dataset
              ensuring quality and balance.
            </p>
          </div>

          <div className="card">
            <h3>Feature Extraction</h3>
            <p>
              Extract both text (tweets, bio) and metadata (followers, friends,
              verification, etc.).
            </p>
          </div>

          <div className="card">
            <h3>Hybrid Model</h3>
            <p>
              Combine BERT embeddings with numerical metadata using deep
              learning for robust classification.
            </p>
          </div>

          <div className="card">
            <h3>Prediction</h3>
            <p>
              Deploy model to classify accounts as Human or Bot in real-time
              with high accuracy.
            </p>
          </div>
        </div>
      </div>

      {/* ✅ Architecture Section */}
      <div className="architecture">
        <h2 className="arch-title">Model Architecture</h2>
        <p className="arch-subtitle">
          The proposed hybrid deep learning architecture integrates both textual and 
          user-level metadata features for robust Twitter bot detection. 
          <br /><br />
          Textual information such as tweets and user bios are first processed using 
          transformer-based embeddings (BERT), capturing contextual semantics and 
          linguistic patterns. In parallel, user metadata — including follower/following 
          ratios, account age, activity frequency, and profile attributes — provides 
          behavioral and structural insights. 
          <br /><br />
          These two feature streams are fused at a dedicated fusion layer, followed by 
          dense and dropout layers to enhance generalization and reduce overfitting. 
          The final classification layer outputs a prediction indicating whether the 
          account is likely a <strong>bot</strong> or a <strong>human</strong>. 
          <br /><br />
          This dual-stream integration ensures the model leverages both 
          <em>linguistic cues</em> and <em>behavioral signals</em>, significantly improving 
          detection accuracy compared to approaches relying on a single feature type.
        </p>

        <div className="arch-flow">
          <div className="arch-block input">Tweets / Bio (Text)</div>
          <div className="arch-block embedding">BERT Embeddings</div>
          <div className="arch-block metadata">User Metadata</div>
          <div className="arch-block fusion">Fusion Layer (Concatenation)</div>
          <div className="arch-block dense">Dense + Dropout Layers</div>
          <div className="arch-block output"> Bot / Human</div>
        </div>
      </div>
      
      {/* ✅ New Research Paper + Conference Section */}
      <div className="architecture">
        <h2 className="arch-title">Resources</h2>

        {/* Conference Website */}
        <h3 style={{ color: "#58a6ff", margin: "40px 0 20px" }}>🌐 Conference Website</h3>
        <div className="pdf-screen">
          <div className="pdf-header">
            <span className="dot red"></span>
            <span className="dot yellow"></span>
            <span className="dot green"></span>
          </div>
          <div className="pdf-display">
            <iframe
              src="https://aitr.ac.in/ICIH-2025/index.html"
              title="ICIH 2025 Conference Website"
              className="pdf-viewer"
            ></iframe>
            <div className="screen-overlay"></div>
          </div>
        </div>

       

        
      </div>
    </div>
  );
}

export default Home;
