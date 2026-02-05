import React from "react";
import "./Models.css";

function Models() {
  return (
    <div className="models-container">
      <h1 className="gradient-text">🧠 Models & Outputs</h1>

      {/* Model Cards */}
      <div className="model-cards">
        <div className="card input">
          <h2>BERT Encoder</h2>
          <p>
            Extracts contextual embeddings from tweets and bios,
            capturing semantic nuances of user content.
          </p>
        </div>
        <div className="card metadata">
          <h2>Metadata Features</h2>
          <p>
            Incorporates follower/following ratios, account age,
            activity frequency, and profile attributes.
          </p>
        </div>
        <div className="card fusion">
          <h2>Hybrid Fusion</h2>
          <p>
            Combines BERT embeddings with metadata features in a
            fusion layer for robust representation learning.
          </p>
        </div>
        <div className="card dense">
          <h2>Dense Layers</h2>
          <p>
            Fully connected layers with dropout and batch normalization
            enhance generalization and reduce overfitting.
          </p>
        </div>
        <div className="card output">
          <h2>Final Prediction</h2>
          <p>
            Outputs whether the account is <b>Human</b> or <b>Bot</b>
            with high accuracy and reliability.
          </p>
        </div>
      </div>

      {/* Results Section */}
      <div className="results-section">
        <h2 className="gradient-text">📊 Experimental Results</h2>
        <p className="desc">
          The Hybrid <b>BERT+Metadata</b> model is evaluated on the{" "}
          <b>TwiBot-20 dataset</b>. Results demonstrate that integrating
          behavioral metadata with semantic embeddings consistently
          outperforms text-only baselines.
        </p>

        <div className="table-wrapper">
          <table className="results-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Encoder</th>
                <th>F1-score</th>
                <th>Accuracy</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>BERT-only</td>
                <td>BERT-base</td>
                <td>0.89</td>
                <td>91.2%</td>
              </tr>
              <tr>
                <td>RoBERTa-only</td>
                <td>RoBERTa-base</td>
                <td>0.91</td>
                <td>92.5%</td>
              </tr>
              <tr>
                <td><b>Hybrid BERT+Metadata ✅</b></td>
                <td>BERT-base + MLP</td>
                <td><b>0.935</b></td>
                <td><b>94.3%</b></td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default Models;
