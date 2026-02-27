from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
import os

load_dotenv()
db_url = os.getenv("DB_URL")

if not db_url:
	raise ValueError("db_url environment variable is not set")

engine = create_engine(db_url)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()