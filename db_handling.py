from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from db_init import Base


class Transactions(Base):
    __tablename__ = "transactions"

    # Primary Key
    id = Column(Integer, primary_key=True, index=True)

    # All numeric columns stored as FLOAT (DOUBLE PRECISION in Postgres)
    flag = Column(Float)

    avg_min_between_sent_tnx = Column(Float)
    avg_min_between_received_tnx = Column(Float)
    time_diff_mins = Column(Float)

    sent_tnx = Column(Float)
    received_tnx = Column(Float)
    number_of_created_contracts = Column(Float)

    max_value_received = Column(Float)
    avg_val_received = Column(Float)
    avg_val_sent = Column(Float)

    total_ether_sent = Column(Float)
    total_ether_balance = Column(Float)

    erc20_total_ether_received = Column(Float)
    erc20_total_ether_sent = Column(Float)
    erc20_total_ether_sent_contract = Column(Float)

    erc20_uniq_sent_addr = Column(Float)
    erc20_uniq_rec_token_name = Column(Float)

    # Text fields
    erc20_most_sent_token_type = Column(String)
    erc20_most_rec_token_type = Column(String)

    # Timestamp
    time = Column(DateTime, default=datetime.utcnow, nullable=False)


class MildlyUnsafeTransaction(Base):
    __tablename__ = "mildly_unsafe_transactions"

    uniq_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    acc_holder = Column(String, index=True)
    time = Column(DateTime, default=datetime.utcnow)