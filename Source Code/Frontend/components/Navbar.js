import React from "react";
import { Link } from "react-router-dom";
import { Home, Info, Cpu, Search, LogIn, UserPlus } from "lucide-react";
import twitterLogo from "./twitter.png";
import "./Navbar.css";

export default function Navbar() {
  return (
    <header className="nav-blur">
      <nav className="ultra-navbar">

        {/* Logo */}
        <Link to="/" className="ultra-logo">
          <img src={twitterLogo} className="ultra-logo-img" />
          <span className="ultra-title">TwitterBot • AI</span>
        </Link>

        {/* Center Links */}
        <ul className="ultra-links">
          <li><Link to="/"><Home size={18}/> Home</Link></li>
          <li><Link to="/about"><Info size={18}/> About</Link></li>
          <li><Link to="/models"><Cpu size={18}/> Models</Link></li>
          <li><Link to="/prediction"><Search size={18}/> Predict</Link></li>
        </ul>

        {/* Buttons */}
        <div className="ultra-actions">
          <Link to="/login" className="btn-glass"><LogIn size={16}/> Login</Link>
          <Link to="/register" className="btn-gradient"><UserPlus size={16}/> Register</Link>
        </div>

      </nav>
    </header>
  );
}
