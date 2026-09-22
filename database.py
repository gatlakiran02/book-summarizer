import sqlite3
from typing import List, Dict

class DatabaseManager:
    def __init__(self, db_path: str = "book_summarizer.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS books (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    title TEXT NOT NULL,
                    author TEXT,
                    chapter TEXT,
                    file_name TEXT,
                    raw_text TEXT NOT NULL,
                    word_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    book_id INTEGER NOT NULL,
                    summary_text TEXT NOT NULL,
                    summary_style TEXT,
                    summary_length TEXT,
                    model_used TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(book_id) REFERENCES books(id)
                )
            """)
            conn.commit()

    def save_book(self, user_id: int, title: str, author: str, chapter: str, file_name: str, raw_text: str) -> int:
        word_count = len(raw_text.split())
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO books (user_id, title, author, chapter, file_name, raw_text, word_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (user_id, title, author, chapter, file_name, raw_text, word_count))
            conn.commit()
            return cursor.lastrowid

    def save_summary(self, book_id: int, summary_text: str, style: str, length: str, model: str) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO summaries (book_id, summary_text, summary_style, summary_length, model_used)
                VALUES (?, ?, ?, ?, ?)
            """, (book_id, summary_text, style, length, model))
            conn.commit()
            return cursor.lastrowid

    def search_books(self, query: str = "") -> List[Dict]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if query:
                pattern = f"%{query}%"
                cursor.execute("""
                    SELECT * FROM books 
                    WHERE title LIKE ? OR author LIKE ? 
                    ORDER BY created_at DESC
                """, (pattern, pattern))
            else:
                cursor.execute("SELECT * FROM books ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]
