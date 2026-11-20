-- CreditLens AI: the core banking stand-in, as Oracle would have it.
--
-- Run this against Oracle Database 23ai Free (Docker) to get the same table the SQLite
-- fallback builds on a laptop:
--
--   docker run -d --name oracle-free -p 1521:1521 -e ORACLE_PWD=creditlens \
--     container-registry.oracle.com/database/free:latest
--   sqlplus system/creditlens@localhost:1521/FREEPDB1 @src/creditlens/integrations/oracle_init.sql
--
-- Then export ORACLE_DSN=localhost:1521/FREEPDB1, ORACLE_USER=creditlens,
-- ORACLE_PASSWORD=creditlens and CoreBankingAdapter uses Oracle instead of SQLite.
--
-- This file is self-contained on purpose: it creates the user, the table, the index and a
-- handful of rows, and it is safe to run twice.

-- 1. A least-privilege user. The agent only ever reads.
DECLARE
  user_exists NUMBER;
BEGIN
  SELECT COUNT(*) INTO user_exists FROM all_users WHERE username = 'CREDITLENS';
  IF user_exists = 0 THEN
    EXECUTE IMMEDIATE 'CREATE USER creditlens IDENTIFIED BY creditlens';
  END IF;
END;
/

GRANT CREATE SESSION TO creditlens;
GRANT CREATE TABLE TO creditlens;
ALTER USER creditlens QUOTA 100M ON users;

-- 2. The legacy table. Note the 1998 column names: this is what the adapter translates.
BEGIN
  EXECUTE IMMEDIATE 'DROP TABLE creditlens.core_loans';
EXCEPTION
  WHEN OTHERS THEN
    IF SQLCODE != -942 THEN RAISE; END IF;  -- -942 is "table does not exist"
END;
/

CREATE TABLE creditlens.core_loans (
  cust_id         VARCHAR2(16)  NOT NULL,
  outstanding_amt NUMBER(14)    NOT NULL,
  product_cd      VARCHAR2(8)   NOT NULL,
  opened_on       DATE,
  CONSTRAINT core_loans_pk PRIMARY KEY (cust_id, product_cd)
);

CREATE INDEX creditlens.core_loans_cust_ix ON creditlens.core_loans (cust_id);

COMMENT ON TABLE creditlens.core_loans IS
  'Outstanding facilities per customer. Amounts are INR thousands.';
COMMENT ON COLUMN creditlens.core_loans.cust_id IS 'Maps to CreditLens company_id';
COMMENT ON COLUMN creditlens.core_loans.outstanding_amt IS 'Maps to CreditLens outstanding_inr_000';
COMMENT ON COLUMN creditlens.core_loans.product_cd IS 'WCDL, TL or EQF. Maps to CreditLens product';

-- 3. A few rows so a query returns something before the full load.
--    The real load comes from data/raw/registry/existing_loans.csv:
--    python -m creditlens.integrations.oracle --rebuild
INSERT INTO creditlens.core_loans (cust_id, outstanding_amt, product_cd, opened_on)
  VALUES ('C001', 91270, 'WCDL', DATE '2024-04-01');
INSERT INTO creditlens.core_loans (cust_id, outstanding_amt, product_cd, opened_on)
  VALUES ('C002', 22120, 'WCDL', DATE '2024-04-01');
INSERT INTO creditlens.core_loans (cust_id, outstanding_amt, product_cd, opened_on)
  VALUES ('C034', 70950, 'TL',   DATE '2023-10-12');

COMMIT;

-- 4. The read-only grant the agent's account needs.
GRANT SELECT ON creditlens.core_loans TO creditlens;
