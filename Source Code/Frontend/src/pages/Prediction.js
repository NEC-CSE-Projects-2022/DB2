import React, { useState } from "react";
import "./Prediction.css";

function Prediction() {
  const [text, setText] = useState("");
  const [followers, setFollowers] = useState("");
  const [friends, setFriends] = useState("");
  const [listed, setListed] = useState("");
  const [statuses, setStatuses] = useState("");
  const [verified, setVerified] = useState(false);

  const [result, setResult] = useState(null);
  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState(false);

  const sanitizeText = (str) => str.replace(/<[^>]+>/g, "").trim();

  // ============================================================
  // VALIDATION LOGIC (REALISTIC + STRICT)
  // ============================================================
  const validate = () => {
    const errs = {};

    // --- BASIC REQUIRED FIELDS ---
    if (!text.trim()) {
      errs.text = "Tweet text is required for semantic analysis.";
    } else if (text.trim().length < 5) {
      errs.text = "Tweet text is too short. Enter a meaningful sentence.";
    }

    if (followers === "") errs.followers = "Followers count is required.";
    if (friends === "") errs.friends = "Friends count is required.";
    if (listed === "") errs.listed = "Listed count is required.";
    if (statuses === "") errs.statuses = "Statuses count is required.";

    // --- CONVERSIONS ---
    const f = Number(followers);
    const fr = Number(friends);
    const l = Number(listed);
    const s = Number(statuses);

    // --- NUMERIC VALIDATION ---
    if (!Number.isFinite(f) || f < 0)
      errs.followers = "Followers must be non-negative.";
    if (!Number.isFinite(fr) || fr < 0)
      errs.friends = "Friends count must be non-negative.";
    if (!Number.isFinite(l) || l < 0)
      errs.listed = "Listed count must be non-negative.";
    if (!Number.isFinite(s) || s < 0)
      errs.statuses = "Statuses count must be non-negative.";

    // --- REALISTIC BEHAVIOR VALIDATIONS ---
    if (f === 0 && fr === 0 && s === 0) {
      errs.realistic = "Metadata looks unrealistic. Please enter real account activity.";
    }

    if (verified && f < 50) {
      errs.verified = "Verified accounts generally have at least 50–100 followers.";
    }

    if (l > f) {
      errs.listed = "Listed count cannot be greater than followers.";
    }

    if (fr > f * 200) {
      errs.friends =
        "Friends count is abnormally high relative to followers. Please recheck.";
    }

    if (text.trim().length < 10 && f < 20 && fr < 20 && s < 20) {
      errs.realistic =
        "Data too weak for meaningful prediction. Provide more realistic account activity.";
    }

    return errs;
  };

  const isFormValid = Object.keys(validate()).length === 0;

  // ============================================================
  // SUBMIT HANDLER
  // ============================================================
  const handleSubmit = async () => {
    const validationErrors = validate();
    setErrors(validationErrors);

    // Stop submission on unrealistic or invalid data
    if (validationErrors.realistic) {
      setResult({ prediction: validationErrors.realistic });
      return;
    }

    if (Object.keys(validationErrors).length > 0) {
      return;
    }

    setLoading(true);
    setResult(null);

    const metadata = {
      followers_count: Number(followers),
      friends_count: Number(friends),
      listed_count: Number(listed),
      statuses_count: Number(statuses),
      verified: verified ? 1 : 0,
    };

    try {
      const res = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: sanitizeText(text), metadata }),
      });

      if (!res.ok) throw new Error("Server error");

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setResult({ prediction: "Error: Could not fetch prediction." });
    } finally {
      setLoading(false);
    }
  };

  // ============================================================
  // RESET HANDLER
  // ============================================================
  const handleReset = () => {
    setText("");
    setFollowers("");
    setFriends("");
    setListed("");
    setStatuses("");
    setVerified(false);
    setErrors({});
    setResult(null);
  };

  // ============================================================
  // UI
  // ============================================================
  return (
    <div className="prediction-container">
      <h1 className="prediction-title">🤖 Twitter Bot Detector</h1>

      <div className="prediction-card">
        <form className="prediction-form" onSubmit={(e) => e.preventDefault()}>
          
          {/* Tweet text */}
          <textarea
            className="prediction-input"
            rows="4"
            placeholder="Enter tweet text..."
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          {errors.text && <p className="error">{errors.text}</p>}

          {/* Followers */}
          <input
            type="number"
            placeholder="Followers Count"
            className="prediction-input"
            value={followers}
            onChange={(e) => setFollowers(e.target.value)}
          />
          {errors.followers && <p className="error">{errors.followers}</p>}

          {/* Friends */}
          <input
            type="number"
            placeholder="Friends Count"
            className="prediction-input"
            value={friends}
            onChange={(e) => setFriends(e.target.value)}
          />
          {errors.friends && <p className="error">{errors.friends}</p>}

          {/* Listed */}
          <input
            type="number"
            placeholder="Listed Count"
            className="prediction-input"
            value={listed}
            onChange={(e) => setListed(e.target.value)}
          />
          {errors.listed && <p className="error">{errors.listed}</p>}

          {/* Statuses */}
          <input
            type="number"
            placeholder="Statuses Count"
            className="prediction-input"
            value={statuses}
            onChange={(e) => setStatuses(e.target.value)}
          />
          {errors.statuses && <p className="error">{errors.statuses}</p>}

          {/* Verified Checkbox */}
          <label className="prediction-checkbox">
            <input
              type="checkbox"
              checked={verified}
              onChange={(e) => setVerified(e.target.checked)}
            />
            <span>Verified Account</span>
          </label>
          {errors.verified && <p className="error">{errors.verified}</p>}

          {/* Buttons */}
          <div className="prediction-buttons">
            <button
              type="button"
              onClick={handleSubmit}
              className="prediction-button"
              disabled={!isFormValid || loading}
            >
              {loading ? "Analyzing..." : "Predict"}
            </button>

            <button
              type="button"
              onClick={handleReset}
              className="prediction-reset"
            >
              Reset
            </button>
          </div>
        </form>

        {/* Prediction Result Display */}
        {result && (
          <div className="prediction-result">
            <p>
              🔍 <b>{result.prediction}</b>
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default Prediction;
