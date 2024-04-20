from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary
from sqlalchemy.orm import sessionmaker

# Configuração do banco de dados SQLite
SQLALCHEMY_DATABASE_URL = "sqlite:///./captcha_data.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)