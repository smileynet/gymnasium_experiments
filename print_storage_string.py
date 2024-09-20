import os
from dotenv import load_dotenv

load_dotenv()

# Fetch database connection details from environment variables
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")

# Construct the PostgreSQL connection URL
mysql_url = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

print("Database connection URL: ", mysql_url)
