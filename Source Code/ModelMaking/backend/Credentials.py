import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
conn = sqlite3.connect("users.db")  # path relative to your Flask app

# ------------------------------
# Database setup
# ------------------------------
DB_NAME = "users.db"

# Create table if not exists
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

# Register user
def register_user(username, password):
    init_db()
    hashed_pw = generate_password_hash(password)
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))
        conn.commit()
        return "Account created successfully!"
    except sqlite3.IntegrityError:
        # Username already exists -> auto-login
        return "Account already exists. Logged in successfully!"
    finally:
        conn.close()

# Login user
def login_user(username, password):
    init_db()
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    if row and check_password_hash(row[0], password):
        return True, "Login successful!"
    return False, "Invalid username or password"

# ------------------------------
# Initialize database automatically
# ------------------------------
init_db()
