import React from "react";
import "./About.css";

function About() {
  return (
    <div className="about-container">
      <div className="about-card">
        <h1 className="about-title">ℹ️ About Project</h1>

        <p>
          This project was developed by our team as part of the final year. It integrates
          machine learning and deep learning to solve real-world social media security challenges.
        </p>

        <p>
          Twitter has become a prominent platform for real-time communication; however, its openness
          also makes it vulnerable to manipulation by automated accounts, commonly known as bots.
          Detecting such accounts is essential to safeguard the authenticity of online discourse.
          This paper presents a Hybrid <b>BERT+Metadata</b> deep learning model that combines semantic
          features from tweet content, extracted using a transformer-based encoder, with behavioral
          attributes such as follower count, posting frequency, and verification status.
        </p>

        <p>
          The model is trained and evaluated on the <b>TwiBot-20</b> benchmark dataset, employing focal loss
          to address class imbalance and AdamW optimization for stable convergence. Experimental
          evaluation shows that the hybrid model achieves <b>94.3% accuracy</b> and an
          <b> F1-score of 0.935</b>, outperforming text-only baselines and demonstrating strong
          classification reliability.
        </p>

        <h2>Index Terms</h2>
        <p>Twitter bot detection, deep learning, BERT, metadata, social media analysis, hybrid model</p>

        <h2>Introduction</h2>
        <p>
          Twitter has emerged as a central platform for real-time communication, public discourse,
          and political engagement. Despite its openness and rapid information dissemination, the
          platform is susceptible to manipulation by automated accounts, commonly referred to as bots.
        </p>

        <h2>Related Work</h2>
        <p>
          Bot detection on Twitter has evolved significantly, transitioning from heuristic-based
          methods to sophisticated deep learning and hybrid frameworks. Early approaches relied on
          surface-level indicators such as posting frequency, account age, and follower-to-following
          ratios. However, the increasing sophistication of spambots has reduced the effectiveness of
          rule-based methods. Deep learning techniques and hybrid frameworks that integrate textual,
          metadata, and network features have now become the backbone of robust Twitter bot detection
          models.
        </p>

        {/* 📑 PDF DISPLAY SECTION */}
        <h2>📑 Research Paper</h2>
        <div className="pdf-screen">
          <div className="pdf-header">
            <span className="dot red"></span>
            <span className="dot yellow"></span>
            <span className="dot green"></span>
          </div>
          <div className="pdf-display">
            <iframe
              src="/TWIBOT_BASE_PAPER__A_.pdf"
              title="Research Paper"
              className="pdf-viewer"
            ></iframe>
            <div className="screen-overlay"></div>
          </div>
        </div>

        {/* ========================= IMAGE GRID SECTION ========================= */}
        <h2>📸 Certificates & Team Gallery</h2>

        <div className="image-grid">
          <img src="/Certificate.jpg" alt="Certificate" />
          <img src="/BashaSir.jpg" alt="Basha Sir" />
          
          <img src="/Mastan.jpg" alt="Mastan" />
          <img src="/Phani.jpg" alt="Phani" />
          <img src="/RizwanaMam.jpg" alt="Rizwana Mam" />
          <img src="/Sireesha.jpg" alt="Sireesha Mam" />
        </div>

      </div>
    </div>
  );
}

export default About;
