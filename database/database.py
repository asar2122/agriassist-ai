import sqlite3
from pathlib import Path


DATABASE_DIRECTORY = Path("database")

DATABASE_PATH = DATABASE_DIRECTORY / "agriassist.db"


def get_connection():

    DATABASE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    return connection


def initialize_database():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_history
        (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            user_message TEXT NOT NULL,

            bot_message TEXT NOT NULL,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    connection.commit()

    connection.close()


def save_message(user_message, bot_message):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO chat_history
        (
            user_message,
            bot_message
        )

        VALUES (?, ?)
        """,

        (
            user_message,
            bot_message,
        ),
    )

    connection.commit()

    connection.close()


def get_chat_history():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            user_message,
            bot_message,
            created_at

        FROM chat_history

        ORDER BY created_at DESC
        """
    )

    history = cursor.fetchall()

    connection.close()

    return history