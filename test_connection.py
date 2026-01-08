# test_connection.py
from src.utils.database_utils import DatabaseConnector

def test_connection():
    try:
        connector = DatabaseConnector()
        connection = connector.connect()
        print("Database connection successful!")
        connector.disconnect()
    except Exception as e:
        print(f"Connection failed: {str(e)}")

if __name__ == "__main__":
    test_connection()