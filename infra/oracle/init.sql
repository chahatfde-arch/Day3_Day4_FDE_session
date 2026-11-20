-- Core banking tables, the way a thirty-year-old system would actually name them.
--
-- Run automatically by the `oracle` service in infra/docker-compose.yml: every .sql
-- file in this folder is executed once, after the database is created.
--
-- The point of this file for the workshop: Oracle calls the company CUST_ID and the
-- exposure OUTSTANDING_AMT, while CreditLens calls them company_id and
-- outstanding_inr_000. Somebody has to write that mapping down, and that somebody is
-- creditlens.integrations.oracle.CoreBankingAdapter. Legacy integration is mostly
-- translating names, units and nulls, not clever code.
--
-- Connect afterwards with:
--   ORACLE_DSN=localhost:1521/FREEPDB1 ORACLE_USER=creditlens ORACLE_PASSWORD=creditlens

-- The image runs this file as SYS in the root container (CDB$ROOT). The CREDITLENS user
-- lives in the pluggable database FREEPDB1, so switch to it first, or the table lands
-- in SYS where the app cannot see it (and ORA-01435 is raised on the next line).
ALTER SESSION SET CONTAINER = FREEPDB1;
ALTER SESSION SET CURRENT_SCHEMA = CREDITLENS;

CREATE TABLE CORE_LOANS (
  CUST_ID          VARCHAR2(10)  NOT NULL,   -- CreditLens company_id, for example C003
  OUTSTANDING_AMT  NUMBER(18, 2) NOT NULL,   -- INR thousands, same unit as the registry
  PRODUCT_CD       VARCHAR2(8)   NOT NULL,   -- TL = term loan, CC = cash credit, OD = overdraft
  CONSTRAINT PK_CORE_LOANS PRIMARY KEY (CUST_ID, PRODUCT_CD)
);

COMMENT ON TABLE  CORE_LOANS                 IS 'Good Bank existing exposure by customer and product';
COMMENT ON COLUMN CORE_LOANS.CUST_ID         IS 'Maps to companies.company_id';
COMMENT ON COLUMN CORE_LOANS.OUTSTANDING_AMT IS 'Maps to existing_loans.outstanding_inr_000';
COMMENT ON COLUMN CORE_LOANS.PRODUCT_CD      IS 'TL, CC or OD';

CREATE INDEX IX_CORE_LOANS_CUST ON CORE_LOANS (CUST_ID);

-- A handful of rows so the adapter has something to read before anyone loads the full
-- registry. The SQLite fallback in creditlens.integrations.oracle builds the same
-- shape from data/raw/registry/existing_loans.csv.
INSERT INTO CORE_LOANS (CUST_ID, OUTSTANDING_AMT, PRODUCT_CD) VALUES ('C001', 91270, 'TL');
INSERT INTO CORE_LOANS (CUST_ID, OUTSTANDING_AMT, PRODUCT_CD) VALUES ('C002', 22120, 'CC');
INSERT INTO CORE_LOANS (CUST_ID, OUTSTANDING_AMT, PRODUCT_CD) VALUES ('C003', 18400, 'TL');
INSERT INTO CORE_LOANS (CUST_ID, OUTSTANDING_AMT, PRODUCT_CD) VALUES ('C003',  6250, 'CC');
INSERT INTO CORE_LOANS (CUST_ID, OUTSTANDING_AMT, PRODUCT_CD) VALUES ('C004', 12900, 'OD');
INSERT INTO CORE_LOANS (CUST_ID, OUTSTANDING_AMT, PRODUCT_CD) VALUES ('C005', 45010, 'TL');

COMMIT;
