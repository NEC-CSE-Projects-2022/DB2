import React, { useState } from "react";
import emailjs from "emailjs-com";
import "./Feedback.css";

function Feedback() {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    message: "",
    rating: 3,
  });

  const [status, setStatus] = useState("");

  const emojis = ["😡", "😕", "😐", "😊", "😍"];

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleRatingChange = (e) => {
    setFormData({ ...formData, rating: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    emailjs
      .send(
        "service_2jd0uzb",
        "template_k447agj",
        formData,
        "rX12RX_fTnYgwx2Bp"
      )
      .then(
        () => {
          setStatus("✅ Feedback sent successfully!");
          setFormData({ name: "", email: "", message: "", rating: 3 });
        },
        () => {
          setStatus("❌ Failed to send feedback.");
        }
      );
  };

  return (
    <div className="feedback-container">
      <h2 className="feedback-title">🌟 Share Your Feedback</h2>

      <div className="feedback-card">
        <form onSubmit={handleSubmit} className="feedback-form">

          <input
            type="text"
            name="name"
            className="feedback-input"
            placeholder="👤 Your Name"
            value={formData.name}
            onChange={handleChange}
            required
          />

          <input
            type="email"
            name="email"
            className="feedback-input"
            placeholder="📧 Your Email"
            value={formData.email}
            onChange={handleChange}
            required
          />

          <textarea
            name="message"
            className="feedback-input"
            placeholder="💬 Your Feedback"
            value={formData.message}
            onChange={handleChange}
            required
          />

          {/* slider */}
          <div className="feedback-rating-row">
            <label>Rate Us:</label>

            <input
              type="range"
              min="1"
              max="5"
              step="1"
              value={formData.rating}
              onChange={handleRatingChange}
              className="feedback-slider"
            />

            <div className="feedback-emoji">
              {emojis[formData.rating - 1]}
            </div>
          </div>

          <button type="submit" className="feedback-submit">
            🚀 Send Feedback
          </button>
        </form>

        {status && <p className="feedback-status">{status}</p>}
      </div>
    </div>
  );
}

export default Feedback;
